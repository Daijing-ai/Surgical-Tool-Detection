import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from cholec80_dataset import Cholec80Dataset, CHOLEC80_TOOLS, CHOLEC80_NUM_CLASSES
from models import ResNetBaseline
import utils

try:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        roc_auc_score,
        average_precision_score,
        roc_curve,
        precision_recall_curve,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_roc_curves(all_labels, all_probs, tool_names, output_path):
    if not SKLEARN_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(tool_names)))
    for i, (tool, color) in enumerate(zip(tool_names, colors)):
        try:
            fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            ax.plot(fpr, tpr, color=color, lw=1.5, label=f'{tool} (AUC={auc:.3f})')
        except ValueError:
            ax.plot([0, 1], [0, 1], color=color, lw=1.5, linestyle='--',
                    label=f'{tool} (AUC=N/A)')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Cholec80 Tool Presence')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path / 'roc_curves.png', dpi=150)
    plt.close(fig)
    print(f"ROC curves saved to {output_path / 'roc_curves.png'}")


def plot_pr_curves(all_labels, all_probs, tool_names, output_path):
    if not SKLEARN_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(tool_names)))
    for i, (tool, color) in enumerate(zip(tool_names, colors)):
        try:
            precision, recall, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
            ap = average_precision_score(all_labels[:, i], all_probs[:, i])
            ax.plot(recall, precision, color=color, lw=1.5, label=f'{tool} (AP={ap:.3f})')
        except ValueError:
            pass
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves - Cholec80 Tool Presence')
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path / 'pr_curves.png', dpi=150)
    plt.close(fig)
    print(f"PR curves saved to {output_path / 'pr_curves.png'}")


def plot_per_class_metrics(results, tool_names, output_path):
    metrics_names = ['precision', 'recall', 'f1', 'auc', 'ap']
    n_tools = len(tool_names)
    n_metrics = len(metrics_names)
    x = np.arange(n_tools)
    width = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    fig, ax = plt.subplots(figsize=(14, 6))
    for j, (metric, color) in enumerate(zip(metrics_names, colors)):
        values = [results[t][metric] for t in tool_names]
        bars = ax.bar(x + j * width, values, width, color=color, label=metric.upper(),
                      edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=5.5, rotation=90)

    ax.set_xlabel('Tool')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics - Cholec80 Tool Presence')
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(tool_names, rotation=30, ha='right')
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path / 'per_class_metrics.png', dpi=150)
    plt.close(fig)
    print(f"Per-class metrics saved to {output_path / 'per_class_metrics.png'}")


def plot_support_distribution(results, tool_names, output_path):
    supports = [results[t]['support'] for t in tool_names]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(tool_names)))
    bars = ax.bar(tool_names, supports, color=colors, edgecolor='white')
    for bar, s in zip(bars, supports):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 50,
                str(s), ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Tool')
    ax.set_ylabel('Positive Samples')
    ax.set_title('Tool Presence Distribution - Test Set')
    ax.set_xticklabels(tool_names, rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig(output_path / 'support_distribution.png', dpi=150)
    plt.close(fig)
    print(f"Support distribution saved to {output_path / 'support_distribution.png'}")



def main(args):
    utils.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {utils.device}")

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_dir)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.model_dir).parent / 'eval_results'

    if not data_dir.exists():
        raise Exception(f"Data dir {data_dir} does not exist")
    if not model_path.exists():
        raise Exception(f"Model path {model_path} does not exist")

    output_path.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = Cholec80Dataset(data_dir, split='test', transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    print(f"Test samples: {len(dataset)}")

    model = ResNetBaseline(backbone='resnet50', num_classes=CHOLEC80_NUM_CLASSES)
    state = torch.load(model_path, map_location=utils.device)
    model.load_state_dict(state)
    model = model.to(utils.device)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(utils.device).float()
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = (all_probs >= 0.5).astype(np.float32)

    print(f"\n{'='*80}")
    print("Cholec80 Tool Presence Baseline - Evaluation Results")
    print(f"{'='*80}")
    print(f"Test samples: {len(dataset)}")

    results = {}
    for i, tool in enumerate(CHOLEC80_TOOLS):
        support = int(all_labels[:, i].sum())
        correct = int((all_preds[:, i] == all_labels[:, i]).sum())
        acc = correct / len(all_labels)

        if SKLEARN_AVAILABLE:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels[:, i], all_preds[:, i], average='binary', zero_division=0
            )
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            except ValueError:
                auc = 0.0
            ap = average_precision_score(all_labels[:, i], all_probs[:, i])
        else:
            precision = recall = f1 = auc = ap = 0.0

        results[tool] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'ap': float(ap),
            'accuracy': float(acc),
            'support': support,
        }

    print(f"\n{'Tool':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'AUC':>10} {'AP':>10} {'Acc':>10} {'Support':>10}")
    print("-" * 95)
    for tool in CHOLEC80_TOOLS:
        r = results[tool]
        print(f"{tool:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['f1']:>10.4f} {r['auc']:>10.4f} {r['ap']:>10.4f} "
              f"{r['accuracy']:>10.4f} {r['support']:>10}")

    if SKLEARN_AVAILABLE:
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='micro', zero_division=0
        )
        mAP = np.mean([results[t]['ap'] for t in CHOLEC80_TOOLS])
        macro_auc = np.mean([results[t]['auc'] for t in CHOLEC80_TOOLS])
    else:
        macro_precision = macro_recall = macro_f1 = 0.0
        micro_precision = micro_recall = micro_f1 = 0.0
        mAP = 0.0
        macro_auc = 0.0

    print("-" * 95)
    print(f"{'Macro Avg':<20} {macro_precision:>10.4f} {macro_recall:>10.4f} "
          f"{macro_f1:>10.4f} {macro_auc:>10.4f} {mAP:>10.4f}")
    print(f"{'Micro Avg':<20} {micro_precision:>10.4f} {micro_recall:>10.4f} "
          f"{micro_f1:>10.4f}")
    print(f"\nmAP: {mAP:.4f}")

    overall_accuracy = float((all_preds == all_labels).mean())
    print(f"Overall accuracy: {overall_accuracy:.4f}")

    summary = {
        'per_class': results,
        'macro_avg': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1': float(macro_f1),
            'auc': float(macro_auc),
        },
        'micro_avg': {
            'precision': float(micro_precision),
            'recall': float(micro_recall),
            'f1': float(micro_f1),
        },
        'mAP': float(mAP),
        'overall_accuracy': float(overall_accuracy),
    }

    with open(output_path / 'eval_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path / 'eval_results.json'}")

    plot_roc_curves(all_labels, all_probs, CHOLEC80_TOOLS, output_path)
    plot_pr_curves(all_labels, all_probs, CHOLEC80_TOOLS, output_path)
    plot_per_class_metrics(results, CHOLEC80_TOOLS, output_path)
    plot_support_distribution(results, CHOLEC80_TOOLS, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to cholec80 dataset root')
    parser.add_argument('--model_dir', required=True, help='Path to saved model checkpoint')
    parser.add_argument('--output', default=None, help='Path to save evaluation results')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gpu', default=1, type=int, help='GPU device id')

    args = parser.parse_args()
    main(args)
