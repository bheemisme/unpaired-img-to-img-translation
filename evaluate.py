import torch
import numpy as np
import os

from torchvision.models import inception_v3
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm
from torchvision.utils import save_image
from torchvision import transforms
from load_data import get_dataloaders
from networks import Generator
from losses import cycle_consistency_loss
from utils import Config


def compute_mifid(test_loader, train_loader, real_loader, generator, device, direction="monet_to_photo"):
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
    inception = inception_v3(pretrained=True, transform_input=False).eval().to(device)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])


    # Extract features
    real_features = []
    fake_features = []
    train_features = []
    with torch.no_grad():
        # Real images (target domain)
        for real in real_loader:
            real = transform(real.to(device))
            real_features.append(inception(real).cpu().numpy())
        # Generated images
        for x in test_loader:
            fake = generator(x.to(device))
            fake = transform(fake)
            fake_features.append(inception(fake).cpu().numpy())
        # Training images (target domain)
        for train_img in train_loader:
            train_img = transform(train_img.to(device))
            train_features.append(inception(train_img).cpu().numpy())

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    train_features = np.concatenate(train_features, axis=0)
    
    
    # Compute FID
    mu_r, mu_g = np.mean(real_features, axis=0), np.mean(fake_features, axis=0)
    sigma_r, sigma_g = np.cov(real_features, rowvar=False), np.cov(fake_features, rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r.dot(sigma_g))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * covmean)
    
     # Compute memorization distance
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(train_features)
    distances, _ = nn.kneighbors(fake_features)
    mem_distance = np.mean(distances)

    # Compute MiFID
    if mem_distance < 1e-6:
        mem_distance = 1
    mifid = fid / mem_distance  # Avoid division by zero
    return mifid, fid, mem_distance


def evaluate_cycle_gan(checkpoint_path=None):
    """
    Evaluate CycleGAN on test set with MiFID, FID, cycle consistency loss, and visualization.

    Args:
        checkpoint_path (str, optional): Path to model checkpoint. If None, use latest.

    Returns:
        dict: Evaluation metrics (mifid_combined, mifid_x, mifid_y, fid_x, fid_y, cycle_loss).
    """
    device = Config.device

    # Initialize models
    generator_x_to_y = Generator().to(device)
    generator_y_to_x = Generator().to(device)

    # Load checkpoint
    # if checkpoint_path is None:
    #     checkpoint_path = os.path.join(Config.checkpoint_dir, f"checkpoint_epoch_{Config.num_epochs}.pth")
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # generator_x_to_y.load_state_dict(checkpoint['generator_x_to_y'])
    # generator_y_to_x.load_state_dict(checkpoint['generator_y_to_x'])

    # Data loaders
    train_loader_x, train_loader_y, test_loader_x, test_loader_y = get_dataloaders()

    # Compute MiFID and FID for both directions
    mifid_y, fid_y, mem_dist_y = compute_mifid(
        test_loader_x, train_loader_y, test_loader_y, generator_x_to_y, device, "monet_to_photo"
    )
    mifid_x, fid_x, mem_dist_x = compute_mifid(
        test_loader_y, train_loader_x, test_loader_x, generator_y_to_x, device, "photo_to_monet"
    )

    # Combine MiFID
    mifid_combined = (mifid_x + mifid_y) / 2

    # Compute cycle consistency loss on test set
    cycle_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for real_x, real_y in zip(test_loader_x, test_loader_y):
            real_x = real_x.to(device)
            real_y = real_y.to(device)
            cycle_loss += cycle_consistency_loss(real_x, real_y, generator_x_to_y, generator_y_to_x).item()
            num_batches += 1
    cycle_loss /= num_batches

    # Save generated images for visualization
    os.makedirs(Config.eval_dir, exist_ok=True)
    with torch.no_grad():
        for i, (real_x, real_y) in enumerate(zip(test_loader_x, test_loader_y)):
            if i >= 5:  # Save 5 images per direction
                break
            real_x = real_x.to(device)
            real_y = real_y.to(device)
            fake_y = generator_x_to_y(real_x)
            fake_x = generator_y_to_x(real_y)
            save_image(fake_y, os.path.join(Config.eval_dir, f"fake_photo_{i}.png"), normalize=True)
            save_image(fake_x, os.path.join(Config.eval_dir, f"fake_monet_{i}.png"), normalize=True)

    # Return metrics
    metrics = {
        "mifid_combined": mifid_combined,
        "mifid_x": mifid_x,
        "mifid_y": mifid_y,
        "fid_x": fid_x,
        "fid_y": fid_y,
        "cycle_loss": cycle_loss
    }
    return metrics

if __name__ == "__main__":
    metrics = evaluate_cycle_gan()
    print("Evaluation Metrics:")
    print(f"Combined MiFID: {metrics['mifid_combined']:.4f}")
    print(f"MiFID (Photo-to-Monet): {metrics['mifid_x']:.4f}")
    print(f"MiFID (Monet-to-Photo): {metrics['mifid_y']:.4f}")
    print(f"FID (Photo-to-Monet): {metrics['fid_x']:.4f}")
    print(f"FID (Monet-to-Photo): {metrics['fid_y']:.4f}")
    print(f"Cycle Consistency Loss: {metrics['cycle_loss']:.4f}")
    
