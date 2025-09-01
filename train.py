import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils import Config
from networks import Generator, Discriminator
from losses import cycle_consistency_loss, adversarial_loss, identity_loss
from load_data import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms


def train_cycle_gan():
    """
    Training loop for CycleGAN.
    """
    # Set device
    device = Config.device

    # Initialize models
    generator_x_to_y = Generator().to(device)
    generator_y_to_x = Generator().to(device)
    discriminator_x = Discriminator().to(device)
    discriminator_y = Discriminator().to(device)

    # Initialize weights
    generator_x_to_y.apply(Generator.init_weights)
    generator_y_to_x.apply(Generator.init_weights)
    discriminator_x.apply(Discriminator.init_weights)
    discriminator_y.apply(Discriminator.init_weights)

    # Optimizers
    g_optimizer = optim.Adam(
        list(generator_x_to_y.parameters()) + list(generator_y_to_x.parameters()),
        lr=Config.learning_rate, betas=(Config.beta1, Config.beta2)
    )
    d_optimizer = optim.Adam(
        list(discriminator_x.parameters()) + list(discriminator_y.parameters()),
        lr=Config.learning_rate, betas=(Config.beta1, Config.beta2)
    )

    # Data loaders
    transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * Config.img_channels, std=[0.5] * Config.img_channels)
    ])
    dataset_x = ImageDataset(Config.x_dir, transform=transform)
    dataset_y = ImageDataset(Config.y_dir, transform=transform)
    
    dataloader_x = DataLoader(
        dataset_x, batch_size=Config.batch_size, shuffle=True,
        num_workers=Config.num_workers, pin_memory=Config.pin_memory
    )
    
    dataloader_y = DataLoader(
        dataset_y, batch_size=Config.batch_size, shuffle=True,
        num_workers=Config.num_workers, pin_memory=Config.pin_memory
    )
    
    # Create checkpoint directory
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(Config.num_epochs):
        for i, (real_x, real_y) in enumerate(zip(dataloader_x, dataloader_y)):
            real_x = real_x.to(device)
            real_y = real_y.to(device)

            # --- Train Discriminators ---
            d_optimizer.zero_grad()

            # Generate fake images
            fake_y = generator_x_to_y(real_x)
            fake_x = generator_y_to_x(real_y)

            # Discriminator losses
            d_x_loss = adversarial_loss(real_x, fake_x, discriminator_x, is_discriminator=True)
            d_y_loss = adversarial_loss(real_y, fake_y, discriminator_y, is_discriminator=True)
            d_loss = (d_x_loss + d_y_loss) * 0.5

            d_loss.backward()
            d_optimizer.step()

            # --- Train Generators ---
            g_optimizer.zero_grad()

            # Generate fake images (recompute for generator training)
            fake_y = generator_x_to_y(real_x)
            fake_x = generator_y_to_x(real_y)

            # Adversarial losses
            g_adv_loss = (
                adversarial_loss(real_x, fake_y, discriminator_y, is_discriminator=False) +
                adversarial_loss(real_y, fake_x, discriminator_x, is_discriminator=False)
            ) * 0.5

            # Cycle consistency loss
            cycle_loss = cycle_consistency_loss(real_x, real_y, generator_x_to_y, generator_y_to_x)

            # Identity loss
            id_loss = identity_loss(real_x, real_y, generator_x_to_y, generator_y_to_x)

            # Total generator loss
            g_loss = g_adv_loss + cycle_loss + id_loss

            g_loss.backward()
            g_optimizer.step()

            # Print progress
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{Config.num_epochs}] Batch [{i}] "
                      f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f} "
                      f"(Adv: {g_adv_loss.item():.4f}, Cycle: {cycle_loss.item():.4f}, "
                      f"Id: {id_loss.item():.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator_x_to_y': generator_x_to_y.state_dict(),
                'generator_y_to_x': generator_y_to_x.state_dict(),
                'discriminator_x': discriminator_x.state_dict(),
                'discriminator_y': discriminator_y.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch + 1
            }, os.path.join(Config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))



if __name__ == "__main__":
    train_cycle_gan()