import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
from src.utils.data_loader import DataLoader
from src.utils.evaluation import calculate_rmse, calculate_mae
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF, SVDRecommender
from src.models.hybrid import HybridRecommender

def train_and_evaluate_models(data_path='data', test_size=0.2, random_state=42):
    """
    Train and evaluate multiple recommendation models.
    
    Args:
        data_path (str): Path to the data directory
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing trained models
    """
    # Load and preprocess the data
    print("Loading and preprocessing data...")
    data_loader = DataLoader(data_path=data_path)
    ratings_df, movies_df, _ = data_loader.load_movielens_data()
    ratings_df, movies_df, _ = data_loader.preprocess_data()
    
    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    train_data, test_data = data_loader.split_data(test_size=test_size, random_state=random_state)
    
    # Create a directory to save the models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize models
    models = {
        'ContentBased': ContentBasedRecommender(),
        'UserBasedCF': UserBasedCF(),
        'ItemBasedCF': ItemBasedCF(),
        'SVD': SVDRecommender(n_factors=50),
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        start_time = time.time()
        
        # Train the model
        model.fit(train_data, movies_df)
        
        # Save the model
        model_path = os.path.join(models_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluate the model
        print(f"Evaluating {name} model...")
        
        # Sample a subset of the test data for faster evaluation
        test_sample = test_data.sample(min(1000, len(test_data)), random_state=random_state)
        
        # Generate predictions
        y_true = test_sample['rating'].values
        y_pred = []
        
        for _, row in test_sample.iterrows():
            try:
                pred = model.predict(row['userId'], row['movieId'])
                y_pred.append(pred)
            except Exception as e:
                print(f"Error predicting rating: {e}")
                y_pred.append(test_sample['rating'].mean())
        
        # Calculate metrics
        rmse = calculate_rmse(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)
        
        # Store results
        training_time = time.time() - start_time
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'Training Time': training_time
        }
        
        print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Training Time: {training_time:.2f}s")
    
    # Create and train a hybrid model
    print("\nTraining Hybrid model...")
    hybrid = HybridRecommender(name="Hybrid")
    
    # Add trained models to the hybrid with weights
    for name, model in models.items():
        if name == 'ContentBased':
            weight = 0.3
        elif name == 'SVD':
            weight = 0.3
        else:
            weight = 0.2
            
        hybrid.add_recommender(model, weight=weight)
    
    # Set the hybrid model as fitted since all its component models are already trained
    hybrid.is_fitted = True
    hybrid.ratings_df = train_data
    hybrid.movies_df = movies_df
    
    # Save the hybrid model
    hybrid_path = os.path.join(models_dir, "Hybrid.joblib")
    joblib.dump(hybrid, hybrid_path)
    print(f"Hybrid model saved to {hybrid_path}")
    
    # Evaluate the hybrid model
    print("Evaluating Hybrid model...")
    
    # Sample a subset of the test data for faster evaluation
    test_sample = test_data.sample(min(1000, len(test_data)), random_state=random_state)
    
    # Generate predictions
    y_true = test_sample['rating'].values
    y_pred = []
    
    for _, row in test_sample.iterrows():
        try:
            pred = hybrid.predict(row['userId'], row['movieId'])
            y_pred.append(pred)
        except Exception as e:
            print(f"Error predicting rating: {e}")
            y_pred.append(test_sample['rating'].mean())
    
    # Calculate metrics
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    
    # Store results
    results['Hybrid'] = {
        'RMSE': rmse,
        'MAE': mae,
        'Training Time': 0  # The hybrid model doesn't need training
    }
    
    print(f"Hybrid - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Add the hybrid model to the models dictionary
    models['Hybrid'] = hybrid
    
    # Plot the results
    plot_results(results)
    
    return models

def plot_results(results):
    """
    Plot the evaluation results.
    
    Args:
        results (dict): Dictionary containing evaluation results
    """
    # Create a directory for plots
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract metrics
    models = list(results.keys())
    rmse_values = [results[model]['RMSE'] for model in models]
    mae_values = [results[model]['MAE'] for model in models]
    
    # Set up the figure
    plt.figure(figsize=(12, 6))
    
    # Plot RMSE
    plt.subplot(1, 2, 1)
    sns.barplot(x=models, y=rmse_values)
    plt.title('RMSE by Model')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    sns.barplot(x=models, y=mae_values)
    plt.title('MAE by Model')
    plt.xlabel('Model')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'model_comparison.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Close the figure
    plt.close()

if __name__ == '__main__':
    train_and_evaluate_models() 