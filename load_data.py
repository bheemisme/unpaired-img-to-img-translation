import torch
import torchvision.transforms as transforms


from PIL import Image
from torch.utils.data import Dataset, DataLoader
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


# Define a simple transform
transform = transforms.Compose(
    [
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

x_loader = DataLoader(
    dataset=ImageDataset(directory=Config.x_dir, transform=transform),
    batch_size=Config.batch_size,
    shuffle=True,
    pin_memory=Config.pin_memory,
    num_workers=Config.num_workers
)
y_loader = DataLoader(
    dataset=ImageDataset(directory=Config.y_dir, transform=transform),
    batch_size=Config.batch_size,
    shuffle=True,
    pin_memory=Config.pin_memory,
    num_workers=Config.num_workers
)

if __name__ == "__main__":
    # download_data()

    images = next(iter(x_loader))
    print(len(images))
    print(images[0].shape)
    # print(images[0][0][0][:10])
    print(images[0][1][:10][:10])
    print(torch.max(images[0][1]))
    print(torch.min(images[0][1]))

    #
