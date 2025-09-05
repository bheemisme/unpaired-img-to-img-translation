import torch
import numpy as np
import os
import argparse

import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm
from torchvision import transforms
from load_data import get_dataloaders
from networks import Generator
from utils import Config


def get_features(dataloader, inception, transform, generator=None):
    features = []
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch.to(Config.device)

            imgs.to(Config.device)
            # If generator provided (e.g. X -> Y)
            if generator is not None:
                imgs = generator(imgs)

            # Extract features
            imgs = transform(imgs)
            feats = inception(imgs)[0].view(imgs.size(0), -1)  # (B, 2048)
            features.append(feats.cpu())
    return torch.cat(features, dim=0)  # (N, 2048)


def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)

    # numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def memorization_score(fake_feats, train_feats):
    fake_norm = F.normalize(fake_feats, dim=1)
    train_norm = F.normalize(train_feats, dim=1)

    cos_sim = fake_norm @ train_norm.T  # (N_fake, N_train)
    cos_dist = 1 - cos_sim
    min_dist, _ = torch.min(cos_dist, dim=1)  # nearest neighbor per fake
    min_dist = min_dist.mean().item()

    if min_dist < 1e-6:
        min_dist = 1

    return min_dist


def compute_mifid(test_loader, train_loader, real_loader, generator):
    """
    Compute MiFID, FID, and memorization distance for one translation direction.

    Args:
        test_loader: DataLoader for test inputs (test_loader_x or test_loader_y).
        train_loader: DataLoader for training images of target domain.
        real_loader: DataLoader for real images of target domain (test_loader_y or test_loader_x).
        generator: Generator (generator_x_to_y or generator_y_to_x).
        device: Device (Config.device).
        direction: "monet_to_photo" or "photo_to_monet".

    Returns:
        tuple: (mifid, fid, mem_distance).
    """

    inception = (
        inception_v3(weights=Inception_V3_Weights.DEFAULT, progress=True)
        .eval()
        .to(Config.device)
    )
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    # Extract features
    real_features = get_features(real_loader, inception, transform)
    gen_features = get_features(test_loader, inception, transform, generator)
    train_features = get_features(train_loader, inception, transform)

    # Compute FID
    mu_r, mu_g = torch.mean(real_features, dim=0), torch.mean(gen_features, dim=0)
    sigma_r, sigma_g = np.cov(real_features, rowvar=False), np.cov(
        gen_features, rowvar=False
    )

    fid = calculate_fid(mu_r, sigma_r, mu_g, sigma_g)
    mem_score = memorization_score(gen_features, train_features)

    mifid = fid / mem_score  # Avoid division by zero
    return mifid, fid, mem_score


def evaluate_cycle_gan(checkpoint_path=None):
    """
    Evaluate CycleGAN on test set with MiFID, FID, cycle consistency loss, and visualization.

    Args:
        checkpoint_path (str, optional): Path to model checkpoint. If None, use latest.

    Returns:
        dict: Evaluation metrics (mifid_combined, mifid_x, mifid_y, fid_x, fid_y, cycle_loss).
    """

    # Initialize models
    generator_x_to_y = Generator().to(Config.device).eval()
    generator_y_to_x = Generator().to(Config.device).eval()

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            Config.checkpoint_dir, f"checkpoint_epoch_{Config.num_epochs}.pth"
        )
    checkpoint = torch.load(checkpoint_path, map_location=Config.device, weights_only=True)
    generator_x_to_y.load_state_dict(checkpoint["generator_x_to_y"])
    generator_y_to_x.load_state_dict(checkpoint["generator_y_to_x"])

    # Data loaders
    x_train_loader, y_train_loader = get_dataloaders(mode="train")
    x_val_loader, y_val_loader = get_dataloaders(mode="val")

    # Compute MiFID and FID for both directions
    mifid_y, fid_y, mem_score_y = compute_mifid(
        x_val_loader, y_train_loader, y_val_loader, generator_x_to_y
    )  # x to y
    mifid_x, fid_x, mem_score_x = compute_mifid(
        y_val_loader, x_train_loader, x_val_loader, generator_y_to_x
    )

    # Combine MiFID
    mifid_combined = (mifid_x + mifid_y) / 2

    # Return metrics
    metrics = {
        "mifid_combined": mifid_combined,
        "mifid_x": mifid_x,
        "mifid_y": mifid_y,
        "fid_y": fid_y,
        "mem_score_y": mem_score_y,
        "fid_x": fid_x,
        "mem_score_x": mem_score_x
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Unpaired Image to Image Translation")
    
    parser.add_argument("-c", "--checkpoint", type=str, help="checkpoint path")
    parser.add_argument("-xtd","--x_train_dir", type=str, default=Config.x_train_dir,
                        help="Path to photo images directory")
    parser.add_argument("-ytd","--y_train_dir", type=str, default=Config.y_train_dir,
                        help="Path to monet images directory")
    parser.add_argument("-xvd","--x_val_dir", type=str, default=Config.x_val_dir,
                        help="Path to photo images directory")
    parser.add_argument("-yvd","--y_val_dir", type=str, default=Config.y_val_dir,
                        help="Path to monet images directory")
    
    parser.add_argument("-tbs","--train_batch_size", type=int, default=Config.train_batch_size,
                        help="Batch size for training")
    parser.add_argument("-vbs","--val_batch_size", type=int, default=Config.val_batch_size,
                        help="Batch size for training")
    args = parser.parse_args()
    
    Config.x_train_dir = args.x_train_dir
    Config.y_train_dir = args.y_train_dir
    Config.x_val_dir = args.x_val_dir
    Config.y_val_dir = args.y_val_dir
    
    Config.train_batch_size = args.train_batch_size
    Config.val_batch_size = args.val_batch_size
    
    metrics = evaluate_cycle_gan(checkpoint_path=args.checkpoint)
    print("Evaluation Metrics:")
    print(f"Combined MiFID: {metrics['mifid_combined']:.4f}")
    print(f"MiFID (Photo-to-Monet): {metrics['mifid_x']:.4f}")
    print(f"MiFID (Monet-to-Photo): {metrics['mifid_y']:.4f}")
    print(f"FID (Photo-to-Monet): {metrics['fid_x']:.4f}")
    print(f"FID (Monet-to-Photo): {metrics['fid_y']:.4f}")

    
if __name__ == "__main__":
    main()