import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from cholec80_dataset import Cholec80Dataset, CHOLEC80_NUM_CLASSES
from models import ResNetBaseline
import utils


def main(args):
    utils.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {utils.device}")

    data_dir = Path(args.data_dir)
    logs_dir = Path(args.logs)
    epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr

    if not data_dir.exists():
        raise Exception(f"Data directory {data_dir} does not exist")

    utils.prepare_logs(logs_dir)

    train_transform = T.Compose([
        T.RandomResizedCrop(512, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Cholec80Dataset(data_dir, split='train', transform=train_transform)
    val_dataset = Cholec80Dataset(data_dir, split='val', transform=val_transform)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)

    model = ResNetBaseline(backbone='resnet50', num_classes=CHOLEC80_NUM_CLASSES)
    model = model.to(utils.device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch
    )

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                'model': 'ResNet50',
                'num_classes': CHOLEC80_NUM_CLASSES,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'optimizer': 'AdamW',
                'weight_decay': 1e-4,
                'scheduler': 'CosineAnnealingLR',
                'loss': 'BCEWithLogitsLoss',
            }
        )
        wandb.watch(model, log='all', log_freq=100)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_loss = float('inf')
    patience = args.patience
    no_improve = 0

    for epoch in range(epochs):
        print(f"\nEPOCH: {epoch+1}/{epochs}")

        train_loss, train_acc = utils.train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, debug=args.debug
        )
        val_loss, val_acc = utils.test_one_epoch(
            model, val_loader, loss_fn, debug=args.debug
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        log_msg = (
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        print(log_msg)

        if use_wandb:
            wandb.log({
                'train/loss': train_loss,
                'train/acc': train_acc,
                'val/loss': val_loss,
                'val/acc': val_acc,
                'lr': scheduler.get_last_lr()[0],
                'epoch': epoch + 1,
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), logs_dir / 'checkpoints' / 'best_model.pt')
            torch.save(optimizer.state_dict(), logs_dir / 'checkpoints' / 'optimizer.pt')
            torch.save(scheduler.state_dict(), logs_dir / 'checkpoints' / 'scheduler.pt')
            if use_wandb:
                wandb.save(str(logs_dir / 'checkpoints' / 'best_model.pt'))
            no_improve = 0
            print(f"  * New best model saved (val_loss={val_loss:.4f})")
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    utils.save_vis(logs_dir, train_losses, train_accs, val_losses, val_accs)

    if use_wandb:
        wandb.finish()

    print(f"\nTraining finished. Best val_loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to cholec80 dataset root')
    parser.add_argument('--logs', required=True, help='Path to save logs/checkpoints')
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--patience', default=10, type=int, help='Early stopping patience')
    parser.add_argument('--gpu', default=1, type=int, help='GPU device id')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', default='cholec80-tool-presence')
    parser.add_argument('--wandb_name', default='step1_1_baseline')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)
