import os

from matplotlib import pyplot as plt

import config


def bar_plot(class_names: list, class_counts: list, split: str) -> None:
    fig, axes = plt.subplots(figsize=(7, 5), dpi=100)
    plt.bar(class_names, height=class_counts)
    plt.xticks(rotation=90)
    for index, data in enumerate(class_counts):
        plt.text(x=index, y=data + 1, s=f"{data}", fontdict=dict(fontsize=10), horizontalalignment='center')
    plt.title('{} Data distribution'.format(split))
    plt.savefig('{}_distribution.png'.format(split))


if __name__ == '__main__':
    """
    Run this script to get bar plots of the dataset. Images will be saved in the data folder. 
     
    """
    path = config.dataset_folder
    for split in os.listdir(path):
        split_dir = os.path.join(path, split)  # ex: train

        class_names, class_counts = [], []
        for class_dir in os.listdir(split_dir):  # ex: beige
            nb_samples = len(os.listdir(os.path.join(split_dir, class_dir)))
            class_names.append(class_dir)
            class_counts.append(nb_samples)

        bar_plot(class_names, class_counts, split)
