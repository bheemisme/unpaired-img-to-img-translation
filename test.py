import os
import torch
import argparse

from tqdm import tqdm
from utils import Config
from networks import Generator
from torchvision.utils import save_image, make_grid
from load_data import get_dataloaders


def generate_and_save_images(checkpoint_path, max_images, prefix, dest="y") -> None:
    """
    Generate images using a generator and save them to a directory.
    """
    # Ensure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)

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
        for i, images in enumerate(tqdm(loader, desc="test images")):
            if i >= max_images:
                break
            images = images.to(Config.device)
            generated_images = generator(images)

            comparision = torch.cat((images, generated_images), dim=3)
            # Save images with normalization to [0, 1] for visualization
            grid = make_grid(comparision, normalize=True, value_range=(-1, 1))

            save_image(
                tensor=grid,
                fp=os.path.join(Config.output_dir, f"{prefix}_{i}.png"),
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
    parser.add_argument("-xsd", "--x_test_dir", type=str, help="path to test images of x domain")
    parser.add_argument(
        "-ysd", "--y_test_dir", type=str, help="path to test images of y domain"
    )
    parser.add_argument("-op","--output_dir", type=str, default=Config.output_dir,
                        help="Directory to save evaluation images")

    
    args = parser.parse_args()
    Config.x_test_dir = args.x_test_dir
    Config.y_test_dir = args.y_test_dir
    Config.output_dir = args.output_dir

    generate_and_save_images(
        checkpoint_path=args.checkpoint,
        max_images=args.max_images,
        prefix=args.prefix,
        dest=args.destination,
    )
    
    print(f"saved images to {Config.output_dir}")



if __name__ == "__main__":
    main()
