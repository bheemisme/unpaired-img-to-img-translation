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
                padding_mode="reflect",  # Reflect padding to reduce boundary artifacts
                bias=True,
            ),
            nn.InstanceNorm2d(in_channels),  # Instance normalization for style transfer
            nn.ReLU(inplace=True),  # Inplace ReLU to save memory
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=True,
            ),
            nn.InstanceNorm2d(in_channels),
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


class Generator(nn.Module):
    """
    CycleGAN generator for unpaired image-to-image translation.
    Architecture: Encoder (downsampling) -> Transformer (residual blocks) -> Decoder (upsampling).
    """

    def __init__(
        self,
        in_channels: int = Config.img_channels,
        out_channels: int = Config.img_channels,
        hidden_dim: int = Config.generator_hidden_dim,
        num_residual_blocks: int = Config.num_residual_blocks,
    ):
        """
        Initialize the CycleGAN generator.

        Args:
            in_channels (int): Number of input channels (default: 3 for RGB).
            out_channels (int): Number of output channels (default: 3 for RGB).
            hidden_dim (int): Base number of filters for convolutions.
            num_residual_blocks (int): Number of residual blocks in the transformer.
        """
        super(Generator, self).__init__()

        # Encoder: Downsampling layers
        self.encoder = nn.Sequential(
            # c7s1-k: 7x7 Conv, stride 1, hidden_dim filters
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
                bias=True,
            ),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # d2k: 4x4 Conv, stride 2, 2*hidden_dim filters
            nn.Conv2d(
                hidden_dim,
                hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
                bias=True,
            ),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            # d4k: 4x4 Conv, stride 2, 4*hidden_dim filters
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
                bias=True,
            ),
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
        )

        # Transformer: Stack of residual blocks
        transformer_blocks = []
        for _ in range(num_residual_blocks):
            transformer_blocks.append(ResidualBlock(hidden_dim * 4))
        self.transformer = nn.Sequential(*transformer_blocks)

        # Decoder: Upsampling layers
        self.decoder = nn.Sequential(
            # u2k: 4x4 Transpose Conv, stride 2, 2*hidden_dim filters
            nn.ConvTranspose2d(
                hidden_dim * 4,
                hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            # uk: 4x4 Transpose Conv, stride 2, hidden_dim filters
            nn.ConvTranspose2d(
                hidden_dim * 2,
                hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # c7s1-out_channels: 7x7 Conv, stride 1, out_channels
            nn.Conv2d(
                hidden_dim,
                out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
                bias=True,
            ),
            nn.Tanh(),  # Output in [-1, 1] for normalized images
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, in_channels, height, width].

        Returns:
            torch.Tensor: Generated image tensor of shape [batch_size, out_channels, height, width].
        """
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x
    
    
    @staticmethod
    def init_weights(m):
        """
        Initialize weights for convolutional and transposed convolutional layers.
        Uses normal initialization (mean=0, std=0.02) as per CycleGAN paper.
        """
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)



if __name__ == "__main__":
    # Example usage for testing the residual block
    # Create a sample input tensor with shape [batch_size, channels, height, width]
    batch_size = Config.batch_size
    channels = Config.img_channels
    img_size = Config.img_size
    sample_input = torch.randn(batch_size, channels, img_size, img_size).to(
        Config.device
    )

    # Initialize the residual block
    # res_block = ResidualBlock(in_channels=channels).to(Config.device)

    # # Forward pass
    # output = res_block(sample_input)
    # print(f"Input shape: {sample_input.shape}")
    # print(f"Output shape: {output.shape}")

     # Initialize the generator and apply weight initialization
    generator = Generator().to(Config.device)
    generator.apply(Generator.init_weights)

    # Forward pass
    output = generator(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    