import os
import argparse
from src.utils.download_data import download_movielens

def setup_environment():
    """
    Set up the environment for the application.
    """
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Download the MovieLens dataset if it doesn't exist
    if not os.path.exists(os.path.join('data', 'movies.csv')) or \
       not os.path.exists(os.path.join('data', 'ratings.csv')):
        print("Downloading MovieLens dataset...")
        download_movielens(size='small')
    else:
        print("MovieLens dataset already exists.")

def train_models():
    """
    Train the recommendation models.
    """
    from src.train_models import train_and_evaluate_models
    
    print("Training recommendation models...")
    models = train_and_evaluate_models()
    print("Models trained and saved.")
    
    return models

def run_app():
    """
    Run the Flask web application.
    """
    from app import app
    
    print("Starting the web application...")
    app.run(debug=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--setup', action='store_true', help='Set up the environment')
    parser.add_argument('--train', action='store_true', help='Train the recommendation models')
    parser.add_argument('--run', action='store_true', help='Run the web application')
    parser.add_argument('--all', action='store_true', help='Perform all actions')
    
    args = parser.parse_args()
    
    if args.all or args.setup:
        setup_environment()
    
    if args.all or args.train:
        train_models()
    
    if args.all or args.run:
        run_app()
    
    # If no arguments are provided, run the web application
    if not (args.setup or args.train or args.run or args.all):
        setup_environment()
        run_app() 