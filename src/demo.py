import os
import pandas as pd
import numpy as np
import joblib
from src.utils.data_loader import DataLoader
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF, SVDRecommender
from src.models.hybrid import HybridRecommender

def load_data():
    """
    Load and preprocess the MovieLens dataset.
    
    Returns:
        tuple: (ratings_df, movies_df)
    """
    data_loader = DataLoader()
    ratings_df, movies_df, _ = data_loader.load_movielens_data()
    ratings_df, movies_df, _ = data_loader.preprocess_data()
    
    return ratings_df, movies_df

def load_or_train_models(ratings_df, movies_df):
    """
    Load trained models or train new ones if they don't exist.
    
    Args:
        ratings_df (DataFrame): DataFrame containing user ratings
        movies_df (DataFrame): DataFrame containing movie details
        
    Returns:
        dict: Dictionary containing trained models
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    models = {}
    model_classes = {
        'ContentBased': ContentBasedRecommender,
        'UserBasedCF': UserBasedCF,
        'ItemBasedCF': ItemBasedCF,
        'SVD': SVDRecommender
    }
    
    # Try to load existing models
    for name, model_class in model_classes.items():
        model_path = os.path.join(models_dir, f"{name}.joblib")
        
        if os.path.exists(model_path):
            print(f"Loading {name} model...")
            models[name] = joblib.load(model_path)
        else:
            print(f"Training {name} model...")
            model = model_class()
            model.fit(ratings_df, movies_df)
            models[name] = model
            
            # Save the model
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
    
    # Create or load hybrid model
    hybrid_path = os.path.join(models_dir, "Hybrid.joblib")
    
    if os.path.exists(hybrid_path):
        print("Loading Hybrid model...")
        models['Hybrid'] = joblib.load(hybrid_path)
    else:
        print("Creating Hybrid model...")
        hybrid = HybridRecommender(name="Hybrid")
        
        # Add trained models to the hybrid with weights
        hybrid.add_recommender(models['ContentBased'], weight=0.3)
        hybrid.add_recommender(models['UserBasedCF'], weight=0.2)
        hybrid.add_recommender(models['ItemBasedCF'], weight=0.2)
        hybrid.add_recommender(models['SVD'], weight=0.3)
        
        # Set the hybrid model as fitted
        hybrid.is_fitted = True
        hybrid.ratings_df = ratings_df
        hybrid.movies_df = movies_df
        
        models['Hybrid'] = hybrid
        
        # Save the hybrid model
        joblib.dump(hybrid, hybrid_path)
        print(f"Hybrid model saved to {hybrid_path}")
    
    return models

def get_user_recommendations(user_id, model, movies_df, n=10):
    """
    Get recommendations for a user.
    
    Args:
        user_id (int): User ID
        model: Recommendation model
        movies_df (DataFrame): DataFrame containing movie details
        n (int): Number of recommendations to generate
        
    Returns:
        DataFrame: DataFrame containing recommended movies
    """
    recommendations = model.recommend(user_id, n_recommendations=n, exclude_rated=True)
    
    # Format the recommendations
    recommendations = recommendations.copy()
    recommendations['genres_str'] = recommendations['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    return recommendations[['movieId', 'title', 'genres_str', 'predicted_rating']]

def print_recommendations(recommendations, title):
    """
    Print recommendations in a formatted way.
    
    Args:
        recommendations (DataFrame): DataFrame containing recommended movies
        title (str): Title for the recommendations
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {movie['title']}")
        print(f"   Genres: {movie['genres_str']}")
        if 'predicted_rating' in movie:
            print(f"   Predicted Rating: {movie['predicted_rating']:.2f} / 5.0")
        print()

def main():
    """
    Main function to demonstrate the recommendation system.
    """
    print("Loading data...")
    ratings_df, movies_df = load_data()
    
    print("\nLoading/training models...")
    models = load_or_train_models(ratings_df, movies_df)
    
    # Get user input
    while True:
        try:
            user_id = int(input("\nEnter a user ID (1-610): "))
            if user_id < 1 or user_id > 610:
                print("User ID must be between 1 and 610.")
                continue
                
            n_recommendations = int(input("Enter the number of recommendations (1-20): "))
            if n_recommendations < 1 or n_recommendations > 20:
                print("Number of recommendations must be between 1 and 20.")
                continue
                
            break
        except ValueError:
            print("Please enter valid numbers.")
    
    # Get recommendations from each model
    for name, model in models.items():
        recommendations = get_user_recommendations(user_id, model, movies_df, n_recommendations)
        print_recommendations(recommendations, f"{name} Recommendations for User {user_id}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main() 