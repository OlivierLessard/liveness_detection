import csv
import os.path
import datetime

import numpy as np

from data.dataloader import get_dataloaders
import matplotlib.pyplot as plt
from model.model import NeuralNetwork
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import seaborn as sn
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader

import config


def train(dataloader, model, loss_fn, optimizer, device) -> float:
    """
    Train for a single epoch. Update the several times with the data batches.

    """
    size = len(dataloader.dataset)
    model.train()
    epoch_loss = 0
    epoch_count = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        epoch_count += 1

        if batch % 20 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return epoch_loss / epoch_count


def validate(dataloader, model, loss_fn, device) -> int:
    """
    Perform one evaluation epoch over the validation set

    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= num_batches
    correct /= size
    print(f"Error: \n Val Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss


def test(test_loader: DataLoader, model, save_dir: str, device: str) -> None:
    """
    Perform one evaluation epoch over the test set

    """
    model.eval()
    classes = test_loader.dataset.classes
    y_pred, y_true = [], []
    show = False
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)  # [B, N]

        y_pred += list(pred.argmax(1).cpu().numpy())
        y_true += list(y.cpu().numpy())

        if show:
            first_predicted, first_actual = classes[pred[0].argmax(0)], classes[y[0]]
            print(f'First Predicted: "{first_predicted}", First Actual: "{first_actual}"')
            plt.figure()
            plt.imshow(torch.permute(X[0], (1, 2, 0)).cpu().numpy())
            plt.show()
            plt.close()

    # precision recall
    precision, recall, fscore, support = score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    with open(os.path.join(save_dir, 'test_metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Class', 'Precision', 'Recall', 'F1', 'Support'])
        for i in range(len(classes)):
            row = [classes[i], str(precision.round(2)[i]), str(recall.round(2)[i]), str(fscore.round(2)[i]),
                   str(support[i])]
            w.writerow(iter(row))
        w.writerow(['Accuracy', str(accuracy)])
        w.writerow(['Balanced Accuracy', str(balanced_accuracy)])

    # build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt=".2g")
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))


def main(save_dir):
    """
    Train a model and save it.

    """
    # dataloaders
    training_loader, validation_loader, test_loader = get_dataloaders(config.input_size, config.batch_size, config.dataset_folder)

    # model
    device = get_device()
    model = get_backbone()
    model.to(device)
    print(f"Using {device} device")

    # optimizing parameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    # training loop
    train_losses, val_losses = {}, {}
    best_val_loss = 9999
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loss = train(training_loader, model, loss_fn, optimizer, device)
        val_loss = validate(validation_loader, model, loss_fn, device)

        # on epoch end
        train_losses[epoch + 1] = train_loss
        val_losses[epoch + 1] = val_loss
        if val_loss < best_val_loss:
            torch.save(model, os.path.join(save_dir, 'best.pt'))
            best_val_loss = val_loss
    print("Done!")

    # saving
    torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))
    print("Saved PyTorch Model State to last.pth")
    save_losses(save_dir, train_losses, val_losses)

    # test the checkpoint model
    print("\n Loading the best model for testing \n-------------------------------")
    model = torch.load(os.path.join(save_dir, 'best.pt'))
    test(test_loader, model, save_dir, device)


def get_backbone():
    """
    Load the model. Define the architecture to use in the config.py file.

    :return: the model
    """
    if config.backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[-1] = nn.Linear(1024, config.nb_classes)
    else:  # basic 3 layer CNN
        model = NeuralNetwork(config.input_size, config.nb_classes)
    return model


def save_losses(path: str, train_losses: dict, val_losses: dict) -> None:
    """
    Save losses to CSV files

    """
    # as csv
    with open(os.path.join(path, 'train_losses.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerows(train_losses.items())
    with open(os.path.join(path, 'val_losses.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerows(val_losses.items())

    # as png
    plt.figure()
    plt.plot(list(train_losses.keys()), list(train_losses.values()), label='train', color='b')
    plt.plot(list(val_losses.keys()), list(val_losses.values()), label='val', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training losses')
    plt.legend()
    plt.savefig(os.path.join(path, "training_losses.png"))
    return None


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device


def get_run_dir() -> str:
    train_dir = r'runs/train'
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    save_dir = os.path.join(train_dir, "{:%B %d %Y, %Hh%M %Ss}".format(datetime.datetime.now()))
    os.mkdir(save_dir)
    return save_dir


if __name__ == '__main__':
    """
    Run this script to train a model using the parameters in config.py
    
    """
    save_dir = get_run_dir()

    main(save_dir)
