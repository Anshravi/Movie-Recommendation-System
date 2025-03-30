import os
import zipfile
import requests
from tqdm import tqdm

def download_movielens(size='small', data_dir='data'):
    """
    Download the MovieLens dataset.
    
    Args:
        size (str): Size of the dataset ('small', 'full')
        data_dir (str): Directory to save the dataset
        
    Returns:
        str: Path to the dataset directory
    """
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Set the URL based on the size
    if size == 'small':
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        dataset_name = 'ml-latest-small'
    elif size == 'full':
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
        dataset_name = 'ml-latest'
    else:
        raise ValueError("Size must be 'small' or 'full'")
    
    # Set the paths
    zip_path = os.path.join(data_dir, f'{dataset_name}.zip')
    dataset_path = os.path.join(data_dir, dataset_name)
    
    # Check if the dataset already exists
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    # Download the dataset
    print(f"Downloading {size} MovieLens dataset...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=1024), total=total_size//1024, unit='KB'):
            if chunk:
                f.write(chunk)
    
    # Extract the dataset
    print(f"Extracting dataset to {data_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Remove the zip file
    os.remove(zip_path)
    
    # Copy the files to the data directory
    for filename in os.listdir(dataset_path):
        if filename.endswith('.csv'):
            src_path = os.path.join(dataset_path, filename)
            dst_path = os.path.join(data_dir, filename)
            
            # Copy the file
            with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
                dst.write(src.read())
    
    print(f"Dataset downloaded and extracted to {data_dir}")
    return dataset_path

if __name__ == '__main__':
    download_movielens(size='small') 