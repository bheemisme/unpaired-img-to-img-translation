import os
import torch
import argparse

from utils import Config
from networks import Generator
from torchvision.utils import save_image
from load_data import get_dataloaders



def generate_and_save_images(checkpoint_path, max_images, prefix, dest="y") -> None:
    """
    Generate images using a generator and save them to a directory.
    """
    # Ensure output directory exists
    os.makedirs(Config.eval_dir, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=Config.device)
    generator = Generator()

    if dest == "y":
        generator.load_state_dict(checkpoint["generator_x_to_y"])
        loader, _ = get_dataloaders(mode="test")

    elif dest == "x":
        generator.load_state_dict(checkpoint["generator_y_to_x"])
        _, loader = get_dataloaders(mode="test")

    # Set generator to evaluation mode
    generator.to(Config.device).eval()

    # Generate and save images
    with torch.no_grad():
        for i, images in enumerate(loader):
            if i >= max_images:
                break
            images = images.to(Config.device)
            generated_images = generator(images)

            # Save images with normalization to [0, 1] for visualization
            save_image(
                generated_images,
                os.path.join(Config.eval_dir, f"{prefix}_{i}.png"),
                normalize=True,
                range=(-1, 1),  # Match normalization from load_data.py
            )


def main():
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Unpaired Image to Image Translation")
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to checkpoiont")
    parser.add_argument(
        "-m", "--max_images", type=int, help="max limit to generate images"
    )
    parser.add_argument("-p", "--prefix", type=str, help="prefix for generated images")
    parser.add_argument(
        "-d", "--destination", type=str, help="which domain images to generate"
    )

    args = parser.parse_args()
    generate_and_save_images(
        checkpoint_path=args.checkpoint,
        max_images=args.max_images,
        prefix=args.prefix,
        dest=args.destination,
    )



if __name__ == "__main__":
    main()
