import warnings

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


def get_dataloaders(size: int, batch_size: int, root: str) -> (DataLoader, DataLoader, DataLoader):
    """

    :param size: input image size (height equals width)
    :param batch_size
    :return: the dataloaders
    """
    training_transforms = transforms.Compose([
        # transforms.RandAugment(4, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Normalize(mean=[0.4344, 0.4025, 0.3941], std=[0.1718, 0.1622, 0.1627]),
    ])
    validation_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4344, 0.4025, 0.3941], std=[0.1718, 0.1622, 0.1627]),
    ])
    testing_transforms = validation_transforms

    train_set = ImageFolder(root=root + '/train', transform=training_transforms)
    val_set = ImageFolder(root=root + '/val', transform=validation_transforms)
    test_set = ImageFolder(root=root + '/test', transform=testing_transforms)

    training_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return training_loader, validation_loader, test_loader
