import os
import subprocess
import sys

def install_dependencies():
    """
    Install the required dependencies.
    """
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed successfully.")

def setup_project():
    """
    Set up the project by creating necessary directories and downloading the dataset.
    """
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Download the MovieLens dataset
    print("Downloading MovieLens dataset...")
    from src.utils.download_data import download_movielens
    download_movielens(size='small')
    print("Dataset downloaded successfully.")

if __name__ == '__main__':
    # Install dependencies
    install_dependencies()
    
    # Set up the project
    setup_project()
    
    print("\nSetup completed successfully!")
    print("\nTo run the application, use the following command:")
    print("python run.py --all")
    print("\nOr to run specific steps:")
    print("python run.py --setup  # Set up the environment")
    print("python run.py --train  # Train the recommendation models")
    print("python run.py --run    # Run the web application") 