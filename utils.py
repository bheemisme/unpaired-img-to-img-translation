import zipfile
import shutil
import torch
from pathlib import Path

class Config:
    """
    Class to store global configuration parameters for the CycleGAN project as class-level attributes.
    All parameters are static and accessible without instantiation.
    """
    # Dataset and DataLoader parameters
    batch_size: int = 1  # Standard for CycleGAN to minimize memory usage
    num_workers: int = 2  # Number of DataLoader workers; adjust based on CPU cores
    pin_memory: bool = torch.cuda.is_available()  # Enable pinned memory if GPU is available
    x_dir: str = "./data/monet_jpg"
    y_dir: str = "./data/photo_jpg"
    
    # image parameters
    img_size: int = 256 # width and height of the image
    img_channels: int = 3 # number of channels in the image
    

    # Device configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Automatically select GPU or CPU

    @classmethod
    def print_config(cls):
        """
        Print all configuration parameters for verification.
        """
        print("CycleGAN Configuration:")
        for attr, value in cls.__dict__.items():
            if not attr.startswith('_'):
                print(f"{attr}: {value}")
                
def extract_zip(zip_path: str, extract_dir: str):
    """
    Extract a zip file to a specified directory.

    Args:
        zip_path (str): Path to the zip file to be extracted.
        extract_dir (str): Directory where the zip contents will be extracted.

    Raises:
        FileNotFoundError: If the zip file does not exist.
        zipfile.BadZipFile: If the zip file is invalid or corrupted.
    """
    # Convert paths to Path objects for robust handling
    zip_path = Path(zip_path) # type: ignore
    extract_dir = Path(extract_dir) # type: ignore

    # Check if the zip file exists
    if not zip_path.exists(): # type: ignore
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
     # Remove the output directory if it exists
    if extract_dir.exists(): # type: ignore
        try:
            shutil.rmtree(extract_dir)
            print(f"Removed existing directory: {extract_dir}")
        except OSError as e:
            raise OSError(f"Failed to remove existing directory {extract_dir}: {e}")

    # Create the extraction directory
    extract_dir.mkdir(parents=True, exist_ok=True) # type: ignore
    
    try:
        # Open and extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Successfully extracted {zip_path} to {extract_dir}")
    except zipfile.BadZipFile as e:
        raise zipfile.BadZipFile(f"Invalid or corrupted zip file: {e}")
    
if __name__ == "__main__":
    # Example usage
    # try:
    #     extract_zip("gan-getting-started.zip", "data")
    # except (FileNotFoundError, zipfile.BadZipFile) as e:
    #     print(f"Error: {e}")
    
    print(Config.print_config())