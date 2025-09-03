import argparse
from utils import Config
from train import train_cycle_gan

def parse_args():
    """
    Parse command-line arguments and update Config.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Unpaired Image to Image Translation")
    parser.add_argument("-xtd","--x_train_dir", type=str, default=Config.x_train_dir,
                        help="Path to Monet images directory")
    parser.add_argument("-ytd","--y_train_dir", type=str, default=Config.y_train_dir,
                        help="Path to photo images directory")
    parser.add_argument("-xvd","--x_val_dir", type=str, default=Config.x_val_dir,
                        help="Path to Monet images directory")
    parser.add_argument("-yvd","--y_val_dir", type=str, default=Config.y_val_dir,
                        help="Path to photo images directory")
    parser.add_argument("-xsd","--x_test_dir", type=str, default=Config.x_test_dir,
                        help="Path to Monet images directory")
    parser.add_argument("-ysd","--y_test_dir", type=str, default=Config.y_test_dir,
                        help="Path to photo images directory")
    
    parser.add_argument("-tbs","--train_batch_size", type=int, default=Config.train_batch_size,
                        help="Batch size for training")
    parser.add_argument("-vbs","--val_batch_size", type=int, default=Config.val_batch_size,
                        help="Batch size for training")
    parser.add_argument("-sbs","--test_batch_size", type=int, default=Config.test_batch_size,
                        help="Batch size for training")
    
    
    parser.add_argument("--img_size", type=int, default=Config.img_size,
                        help="Image size (height and width)")
    parser.add_argument("--num_epochs", type=int, default=Config.num_epochs,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=Config.learning_rate,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--beta1", type=float, default=Config.beta1,
                        help="Beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=Config.beta2,
                        help="Beta2 for Adam optimizer")
    parser.add_argument("--lambda_cycle", type=float, default=Config.lambda_cycle,
                        help="Weight for cycle consistency loss")
    parser.add_argument("--lambda_identity", type=float, default=Config.lambda_identity,
                        help="Weight for identity loss")
    
    parser.add_argument("--checkpoint_dir", type=str, default=Config.checkpoint_dir,
                        help="Directory to save checkpoints")
    parser.add_argument("--eval_dir", type=str, default=Config.eval_dir,
                        help="Directory to save evaluation images")
    parser.add_argument("--num_workers", type=int, default=Config.num_workers,
                        help="Number of DataLoader workers")
    parser.add_argument("--pin_memory", type=bool, default=Config.pin_memory,
                        help="Use pin_memory in DataLoader")
    args = parser.parse_args()

    # Update Config with command-line arguments
    Config.x_train_dir = args.x_train_dir
    Config.y_train_dir = args.y_train_dir
    Config.x_val_dir = args.x_val_dir
    Config.y_val_dir = args.y_val_dir
    Config.x_test_dir = args.x_test_dir
    Config.y_test_dir = args.y_test_dir
    
    Config.train_batch_size = args.train_batch_size
    Config.val_batch_size = args.val_batch_size
    Config.test_batch_size = args.test_batch_size
    Config.img_size = args.img_size
    
    Config.num_epochs = args.num_epochs
    Config.learning_rate = args.learning_rate
    Config.beta1 = args.beta1
    Config.beta2 = args.beta2
    Config.lambda_cycle = args.lambda_cycle
    Config.lambda_identity = args.lambda_identity
    
    Config.checkpoint_dir = args.checkpoint_dir
    Config.eval_dir = args.eval_dir
    Config.num_workers = args.num_workers
    Config.pin_memory = args.pin_memory

    return args

def main():
    """
    Main function to run CycleGAN training.
    """
    # Parse arguments and update Config
    parse_args()
    
    Config.print_config()
    print("\n")
    # Call training function
    print(f"Starting CycleGAN training with {Config.num_epochs} epochs...")
    train_cycle_gan()
    print("Training completed.")

if __name__ == "__main__":
    main()