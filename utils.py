import numpy as np
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')


def calc_accuracy(y, y_hat):
    batch_size, num_classes = y_hat.shape
    running_avg = 0.
    for i in range(batch_size):
        predictions = torch.round(torch.sigmoid(y[i, :]))
        running_avg += (torch.sum(predictions == y_hat[i, :])) / num_classes
    return (running_avg / batch_size).item()


def train_one_epoch(model, dataloader, loss, optim, scheduler, debug=False):
    model.train()
    running_loss = 0.
    running_acc = 0.

    print("Training")
    for i, (x, y_hat) in enumerate(tqdm(dataloader)):
        x = x.to(device).float()
        y_hat = y_hat.to(device).float()

        optim.zero_grad()
        y = model(x)
        loss_val = loss(y, y_hat)
        loss_val.backward()
        optim.step()
        scheduler.step()
        running_loss += loss_val.item()
        running_acc += calc_accuracy(y, y_hat)

    return running_loss / (i + 1), running_acc / (i + 1)


def test_one_epoch(model, dataloader, loss, debug=False):
    model.eval()
    running_loss = 0.
    running_acc = 0.

    print("Testing")
    with torch.no_grad():
        for i, (x, y_hat) in enumerate(tqdm(dataloader)):
            x = x.to(device).float()
            y_hat = y_hat.to(device).float()
            y = model(x)
            loss_val = loss(y, y_hat)
            running_loss += loss_val.item()
            running_acc += calc_accuracy(y, y_hat)

    return running_loss / (i + 1), running_acc / (i + 1)


def log_results(train_loss, train_acc, test_loss, test_acc, train_results, test_results, debug=False):
    print(f'Training Loss: {train_results[0]} Testing Loss: {test_results[0]}')
    print(f'Training Accuracy: {train_results[1]} Testing Accuracy: {test_results[1]}')
    train_loss.append(train_results[0])
    train_acc.append(train_results[1])
    test_loss.append(test_results[0])
    test_acc.append(test_results[1])


def prepare_logs(logs_dir):
    if not (logs_dir / 'checkpoints').exists():
        os.mkdir(str(logs_dir / 'checkpoints'))
    if not (logs_dir / 'visualizations').exists():
        os.mkdir(str(logs_dir / 'visualizations'))
    if not (logs_dir / 'metrics').exists():
        os.mkdir(str(logs_dir / 'metrics'))


def save_vis(logs_dir, train_loss, train_acc, val_loss, val_acc):
    np.save(str(logs_dir / 'metrics' / 'training_loss.npy'), np.array(train_loss))
    np.save(str(logs_dir / 'metrics' / 'training_acc.npy'), np.array(train_acc))
    np.save(str(logs_dir / 'metrics' / 'val_loss.npy'), np.array(val_loss))
    np.save(str(logs_dir / 'metrics' / 'val_acc.npy'), np.array(val_acc))

    plt.figure()
    plt.plot(np.array(train_loss))
    plt.title('Training loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.savefig(str(logs_dir / 'visualizations' / 'training_loss.png'))

    plt.figure()
    plt.plot(np.array(train_acc))
    plt.title('Training Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.savefig(str(logs_dir / 'visualizations' / 'training_acc.png'))

    plt.figure()
    plt.plot(np.array(val_loss))
    plt.title('Validation loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.savefig(str(logs_dir / 'visualizations' / 'val_loss.png'))

    plt.figure()
    plt.plot(np.array(val_acc))
    plt.title('Validation Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig(str(logs_dir / 'visualizations' / 'val_acc.png'))
