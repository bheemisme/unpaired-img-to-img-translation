import torch
import torch.optim as optim
import os

from typing import Tuple
from torch import nn
from evaluate import evaluate_cycle_gan
from utils import Config
from networks import Generator, Discriminator
from losses import cycle_consistency_loss, adversarial_loss, identity_loss
from load_data import get_dataloaders



def initialize_models() -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    Initialize CycleGAN generators and discriminators with weight initialization.

    Returns:
        Tuple[nn.Module, nn.Module, nn.Module, nn.Module]: (gen_x_to_y, gen_y_to_x, disc_x, disc_y)
    """
    # Initialize generators
    gen_x_to_y = Generator().to(Config.device)
    gen_y_to_x = Generator().to(Config.device)
    gen_x_to_y.apply(Generator.init_weights)
    gen_y_to_x.apply(Generator.init_weights)

    # Initialize discriminators
    disc_x = Discriminator().to(Config.device)
    disc_y = Discriminator().to(Config.device)
    disc_x.apply(Discriminator.init_weights)
    disc_y.apply(Discriminator.init_weights)

    return gen_x_to_y, gen_y_to_x, disc_x, disc_y


def save_checkpoint(
    epoch: int,
    gen_x_to_y: nn.Module,
    gen_y_to_x: nn.Module,
    disc_x: nn.Module,
    disc_y: nn.Module,
    optimizers: dict,
    filename: str,
) -> None:
    """
    Save model checkpoint.

    Args:
        epoch (int): Current epoch number.
        gen_x_to_y (nn.Module): Generator X to Y.
        gen_y_to_x (nn.Module): Generator Y to X.
        disc_x (nn.Module): Discriminator for domain X.
        disc_y (nn.Module): Discriminator for domain Y.
        optimizers (dict): Dictionary of optimizers.
        filename (str): Path to save the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "generator_x_to_y": gen_x_to_y.state_dict(),
        "generator_y_to_x": gen_y_to_x.state_dict(),
        "discriminator_x": disc_x.state_dict(),
        "discriminator_y": disc_y.state_dict(),
        "opt_gen": optimizers["opt_gen"].state_dict(),
        "opt_disc": optimizers["opt_disc"].state_dict(),
    }
    torch.save(checkpoint, filename)


def train_cycle_gan():
    """
    Training loop for CycleGAN.
    """

    gen_x_to_y, gen_y_to_x, disc_x, disc_y = initialize_models()

    # Optimizers
    g_optimizer = optim.Adam(
        list(gen_x_to_y.parameters()) + list(gen_y_to_x.parameters()),
        lr=Config.learning_rate,
        betas=(Config.beta1, Config.beta2),
    )
    d_optimizer = optim.Adam(
        list(disc_x.parameters()) + list(disc_y.parameters()),
        lr=Config.learning_rate,
        betas=(Config.beta1, Config.beta2),
    )

    # Data loaders
    x_train_loader,y_train_loader  = get_dataloaders(mode='train')

    # Create checkpoint directory
    os.makedirs(Config.checkpoint_dir, exist_ok=True)

    # Create checkpoint directory
    os.makedirs(Config.checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(Config.num_epochs):
        for i, (real_x, real_y) in enumerate(zip(x_train_loader, y_train_loader)):
            real_x = real_x.to(Config.device)
            real_y = real_y.to(Config.device)

            disc_x.train()
            disc_y.train()
            gen_x_to_y.train()
            gen_x_to_y.train()
            
            # --- Train Discriminators ---
            d_optimizer.zero_grad()

            # Generate fake images
            fake_y = gen_x_to_y(real_x)
            fake_x = gen_y_to_x(real_y)

            # Discriminator losses
            d_x_loss = adversarial_loss(
                real_x, fake_x, disc_x, is_discriminator=True
            )
            d_y_loss = adversarial_loss(
                real_y, fake_y, disc_y, is_discriminator=True
            )
            d_loss = (d_x_loss + d_y_loss) * 0.5

            d_loss.backward()
            d_optimizer.step()

            # --- Train Generators ---
            g_optimizer.zero_grad()

            # Generate fake images (recompute for generator training)
            fake_y = gen_x_to_y(real_x)
            fake_x = gen_y_to_x(real_y)

            # Adversarial losses
            g_adv_loss = (
                adversarial_loss(
                    real_x, fake_y, disc_y, is_discriminator=False
                )
                + adversarial_loss(
                    real_y, fake_x, disc_x, is_discriminator=False
                )
            ) * 0.5

            # Cycle consistency loss
            cycle_loss = cycle_consistency_loss(
                real_x, real_y, gen_x_to_y, gen_y_to_x
            )

            # Identity loss
            id_loss = identity_loss(real_x, real_y, gen_x_to_y, gen_y_to_x)

            # Total generator loss
            g_loss = g_adv_loss + cycle_loss + id_loss

            g_loss.backward()
            g_optimizer.step()

            # Print progress for every 100 batches
            if i+1 % 100 == 0 or i+1 == len(x_train_loader) :
                print(
                    f"Epoch [{epoch+1}/{Config.num_epochs}] Batch [{i+1}] "
                    f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f} "
                    f"(Adv: {g_adv_loss.item():.4f}, Cycle: {cycle_loss.item():.4f}, "
                    f"Id: {id_loss.item():.4f})"
                )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch + 1 == Config.num_epochs:
            checkpoint_path = os.path.join(
                Config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )

            save_checkpoint(
                epoch,
                gen_x_to_y,
                gen_y_to_x,
                disc_x,
                disc_y,
                {"opt_gen": g_optimizer, "opt_disc": d_optimizer},
                checkpoint_path,
            )

            # Evaluate on test set
            metrics = evaluate_cycle_gan(checkpoint_path=checkpoint_path)
            print(f"Epoch [{epoch+1}] Evaluation Metrics: {metrics}")



if __name__ == "__main__":
    train_cycle_gan()
