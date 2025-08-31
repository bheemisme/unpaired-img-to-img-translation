import torch
import torch.nn as nn

from utils import Config

class ResidualBlock(nn.Module):
    """
    A residual block for the CycleGAN generator, consisting of two convolutional layers
    with instance normalization and ReLU activation, plus a skip connection.
    """
    def __init__(self, in_channels: int):
        """
        Initialize the residual block.

        Args:
            in_channels (int): Number of input channels (matches output channels).
        """
        super(ResidualBlock, self).__init__()
        
        # Define the residual block architecture
        # Conv -> InstanceNorm -> ReLU -> Conv -> InstanceNorm
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='reflect',  # Reflect padding to reduce boundary artifacts
                bias=True
            ),
            nn.InstanceNorm2d(in_channels),  # Instance normalization for style transfer
            nn.ReLU(inplace=True),  # Inplace ReLU to save memory
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='reflect',
                bias=True
            ),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block, adding the input to the output (skip connection).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        # Add input to the output of the block (residual connection)
        return x + self.block(x)


if __name__ == "__main__":
    # Example usage for testing the residual block
    # Create a sample input tensor with shape [batch_size, channels, height, width]
    batch_size = Config.batch_size
    channels = Config.img_channels
    img_size = Config.img_size
    sample_input = torch.randn(batch_size, channels, img_size, img_size).to(Config.device)

    # Initialize the residual block
    res_block = ResidualBlock(in_channels=channels).to(Config.device)

    # Forward pass
    output = res_block(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")