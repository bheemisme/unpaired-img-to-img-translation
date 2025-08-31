# Unpaired Image to Image Translation

This repository contains the implementation of an unpaired image-to-image translation project using a Cycle-Consistent Generative Adversarial Network (CycleGAN) in Python with PyTorch. The project aims to translate images between two domains (e.g., horses to zebras, monet to photo) without paired data, leveraging CycleGAN's ability to learn mappings through adversarial and cycle consistency losses.

## Project Status

The project is under active development. Below is a summary of the implemented components as of August 31, 2025:

1. **Data Extraction Utility**:

   - Added a utility function to extract zip files into a specified directory, with the option to remove the output directory if it exists. This facilitates dataset preparation for domains X and Y.

2. **Global Configuration Class**:

   - Implemented a `CycleGANConfig` class to store global parameters (e.g., `batch_size`, `device`, `img_size`) as class-level attributes. This centralizes configuration for consistent use across the project.

3. **Custom Image Dataset Module**:

   - Created a custom PyTorch `Dataset` class (`ImageDataset`) to load images from a directory with optional transforms. It supports RGB images and is designed for loading datasets for domains X and Y.

4. **DataLoaders for Domains X and Y**:

   - Initialized PyTorch `DataLoader` instances for both domains (X and Y), configured with parameters like `batch_size=1`, `num_workers`, and `pin_memory` for efficient data loading, tailored to CycleGAN's requirements.

5. **Generator Residual Block**:

   - Implemented a `ResidualBlock` as a custom PyTorch `nn.Module`, a core component of the CycleGAN generator. It includes two convolutional layers with instance normalization, ReLU activation, and reflection padding, plus a skip connection.

6. **Generator**:

   - Implemented a `Generator` as a custom PyTorch `nn.Module`

7. **Discriminator**

   - Implemented a `Discriminator` as a custom PyTorch `nn.Module`

## Next Steps

- Develop the training loop, including adversarial, cycle consistency, and identity losses.
- Add utilities for saving and visualizing generated images.
- Test and validate the model on the dataset.

## Repository Structure

- `utils.py`: All necessary utilities are implemented

- `load_data.py`: Logic for loading data into dataloaders

- `networks.py`: Contains code for generator and discriminator.
