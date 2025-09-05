import torch
import numpy as np
import os
import argparse

import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg
from torchvision import transforms
from load_data import get_dataloaders
from networks import Generator
from utils import Config

# Load InceptionV3 once and return
def get_inception_model():
    model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
    model.fc = torch.nn.Identity()   # remove classification head, use 2048-D features
    model = model.to(Config.device).eval()
    return model

@torch.no_grad()
def get_features(loader, model, generator=None):
    feats = []
    for batch in loader:
        # handle (images, labels) or just images
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(Config.device, dtype=torch.float32)
          # If generator is given, map X → Y
        if generator is not None:
            imgs = generator(imgs)
            
        # rescale [-1,1] → [0,1]
        imgs = (imgs + 1) / 2.0
        imgs = torch.clamp(imgs, 0, 1)

        # resize for inception
        imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)

        # forward pass
        f = model(imgs)   # (B, 2048)
        feats.append(f.cpu().numpy())

    return np.concatenate(feats, axis=0)  # shape (N, 2048)


# Compute mean and covariance
def compute_stats(feats: np.ndarray):
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma
# Compute FID given stats
def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    cov_prod = sigma1.dot(sigma2)
    covmean, _ = linalg.sqrtm(cov_prod, disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


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
  
    # Extract features
    real_features = get_features(real_loader, inception)
    gen_features = get_features(test_loader, inception, generator)
    train_features = get_features(train_loader, inception)

    # Compute FID
    mu_r, sigma_r = compute_stats(real_features)
    mu_g, sigma_g = compute_stats(gen_features)
   

    fid = compute_fid(mu_r, sigma_r, mu_g, sigma_g)
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
    print(f"Mi (Photo-to-Monet): {metrics['mem_score_x']:.4f}")
    print(f"Mi (Monet-to-Photo): {metrics['mem_score_y']:.4f}")


    
if __name__ == "__main__":
    main()