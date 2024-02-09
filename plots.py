import os

import cv2
import torch
from matplotlib import pyplot as plt

from data.dataloader import get_dataloaders
import config
from train import get_device, get_backbone
import argparse


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="runs/train/February 12 2024, 17h23 09s/best.pt",
                        help="Set the model path for loading")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def subplot(images: list, preds: list, labels: list, classes: list, save_dir: str, batch_id: int):
    """
    Save a subplot of 24 images from the batch.

    """
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(8, 8))
    i = 0
    for image, prediction, label in zip(images, preds, labels):
        row = i // 6
        col = i % 6
        axes[row, col].imshow(image)
        axes[row, col].set_axis_off()
        axes[row, col].set_title(f"Pred: {classes[prediction]}\nTrue:{classes[label]}", fontsize=10)
        if prediction != label:
            axes[row, col].title.set_color('red')
        i += 1
        if i == 4 * 6:  # because rows=4, cols=6
            break
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(save_dir, f"test_subplot_{batch_id}.png"))
    print(f" saving test_subplot_{batch_id}.png")


if __name__ == '__main__':
    """
    Plot test predictions using a trained model. 
    
    """
    opt = parse_opt()

    # save dir
    save_dir = os.path.join(os.path.split(opt.model_path)[0], 'test_subplots')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # dataloaders
    training_loader, validation_loader, test_loader = get_dataloaders(config.input_size, config.batch_size,
                                                                      config.dataset_folder)

    # model
    device = get_device()
    model = get_backbone()
    model.to(device)
    print(f"Using {device} device")
    model = torch.load(opt.model_path)

    # test loop for plotting
    model.eval()
    y_pred = []
    samples, classes = test_loader.dataset.samples, test_loader.dataset.classes

    for batch_id, (X, y) in enumerate(test_loader):
        # inference on the preprocessed images
        X, y = X.to(device), y.to(device)
        pred = model(X)  # [B, N]
        y_pred = list(pred.argmax(1).cpu().numpy())

        # get the original batch images
        images = []
        labels = []
        for j in range(X.shape[0]):
            path, label = samples[batch_id * test_loader.batch_size + j]
            raw_image = cv2.imread(path)
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            images.append(raw_image)
            labels.append(label)

        # subplots of 24 samples TODO plot the remaining samples of the batch
        subplot(images, y_pred, labels, classes, save_dir, batch_id)
