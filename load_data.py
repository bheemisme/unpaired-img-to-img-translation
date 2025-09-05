import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from pathlib import Path
from utils import Config


class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images from a directory.

    Args:
        directory (str): Path to the directory containing images.
        transform (callable, optional): Optional transform to be applied to images.
    """

    def __init__(self, directory: str, transform=None):
        """
        Initialize the dataset.

        Args:
            directory (str): Path to the directory containing images.
            transform (callable, optional): Transform to apply to each image.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # List of supported image extensions
        self.image_extensions = (".jpg", ".jpeg")

        # Collect all image files in the directory
        self.image_paths = [
            p
            for p in self.directory.iterdir()
            if p.suffix.lower() in self.image_extensions
        ]

        self.transform = transform

    def __len__(self) -> int:
        """
        Return the total number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get an image from the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: Transformed image tensor.

        Raises:
            FileNotFoundError: If the image file cannot be opened.
        """
        img_path = self.image_paths[idx]
        try:
            # Load image using PIL (RGB mode)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image {img_path}: {e}")

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        return image  # type: ignore


def make_loaders(x_ds, y_ds):
    x_sampler = RandomSampler(
        data_source=x_ds,
        replacement=True,
        num_samples=min(len(x_ds), len(y_ds)),
    )
    y_sampler = RandomSampler(
        data_source=y_ds,
        replacement=True,
        num_samples=max(len(x_ds), len(y_ds)),
    )

    x_loader = DataLoader(
        x_ds,
        batch_size=Config.train_batch_size,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=True,
        sampler=x_sampler,
    )
    y_loader = DataLoader(
        y_ds,
        batch_size=Config.train_batch_size,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
        drop_last=True,
        sampler=y_sampler,
    )

    return x_loader, y_loader


def get_dataloaders(mode="train"):
    """
    Create training and test DataLoaders for both domains (X: Monet, Y: photos).

    Returns:
        tuple: (train_loader_x, train_loader_y, test_loader_x, test_loader_y)
    """
    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5] * Config.img_channels, std=[0.5] * Config.img_channels
            ),
        ]
    )

    match mode:
        case "train":
            x_dataset = ImageDataset(Config.x_train_dir, transform=transform)
            y_dataset = ImageDataset(Config.y_train_dir, transform=transform)
        case "val":
            x_dataset = ImageDataset(Config.x_val_dir, transform=transform)
            y_dataset = ImageDataset(Config.y_val_dir, transform=transform)
        case "test":
            x_dataset = ImageDataset(Config.x_test_dir, transform=transform)
            y_dataset = ImageDataset(Config.y_test_dir, transform=transform)

    x_loader, y_loader = make_loaders(x_dataset, y_dataset)

    return (
        x_loader,
        y_loader,
    )


if __name__ == "__main__":
    # download_data()

    # Define a simple transform
    transform = transforms.Compose(
        [
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    x_loader, y_loader = get_dataloaders(mode="train")

    # count = 0
    # for imgx, imgy in zip(x_loader, y_loader):
    #     count += 1

    # print(count)

    # images = next(iter(x_loader))
    # print(len(images))
    # print(images[0].shape)
    # # print(images[0][0][0][:10])
    # print(images[0][1][:10][:10])
    # print(torch.max(images[0][1]))
    # print(torch.min(images[0][1]))
    print(len(x_loader))

    #
