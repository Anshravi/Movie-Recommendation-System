import numpy as np
import pandas as pd
from .base_model import BaseRecommender
from .content_based import ContentBasedRecommender
from .collaborative_filtering import UserBasedCF, ItemBasedCF, SVDRecommender

class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender that combines multiple recommendation approaches.
    
    This model combines the predictions from multiple recommenders to generate more accurate recommendations.
    """
    
    def __init__(self, name="Hybrid", weights=None):
        """
        Initialize the hybrid recommender.
        
        Args:
            name (str): Name of the recommender
            weights (dict, optional): Dictionary mapping recommender names to weights
        """
        super().__init__(name)
        self.recommenders = {}
        self.weights = weights or {}
        self.ratings_df = None
        self.movies_df = None
        
    def add_recommender(self, recommender, weight=1.0):
        """
        Add a recommender to the hybrid model.
        
        Args:
            recommender (BaseRecommender): Recommender to add
            weight (float): Weight of the recommender in the hybrid model
        """
        self.recommenders[recommender.name] = recommender
        self.weights[recommender.name] = weight
        
    def fit(self, ratings_df, movies_df=None):
        """
        Train all the recommenders in the hybrid model.
        
        Args:
            ratings_df (DataFrame): DataFrame containing user ratings
            movies_df (DataFrame, optional): DataFrame containing movie details
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Train each recommender
        for name, recommender in self.recommenders.items():
            print(f"Training {name} recommender...")
            recommender.fit(ratings_df, movies_df)
            
        self.is_fitted = True
        
        return self
    
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a user-item pair.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
            
        Returns:
            float: Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        if not self.recommenders:
            raise ValueError("No recommenders added to the hybrid model.")
            
        # Get predictions from each recommender
        predictions = {}
        total_weight = 0
        
        for name, recommender in self.recommenders.items():
            try:
                pred = recommender.predict(user_id, movie_id)
                weight = self.weights.get(name, 1.0)
                predictions[name] = (pred, weight)
                total_weight += weight
            except Exception as e:
                print(f"Error getting prediction from {name}: {e}")
                
        if not predictions:
            return self.ratings_df['rating'].mean()
            
        # Calculate the weighted average prediction
        weighted_sum = sum(pred * weight for pred, weight in predictions.values())
        weighted_avg = weighted_sum / total_weight
        
        # Clip the rating to be within the valid range
        return max(0.5, min(5.0, weighted_avg))
    
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations (int): Number of recommendations to generate
            exclude_rated (bool): Whether to exclude items the user has already rated
            
        Returns:
            DataFrame: DataFrame containing recommended movies with their details
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        if not self.recommenders:
            raise ValueError("No recommenders added to the hybrid model.")
            
        # Get the movies the user has already rated
        if exclude_rated and self.ratings_df is not None:
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            rated_movies = user_ratings['movieId'].values
        else:
            rated_movies = []
            
        # Get all movies
        if self.movies_df is not None:
            all_movies = self.movies_df['movieId'].values
        else:
            # If movies_df is not available, use the movies in the ratings data
            all_movies = self.ratings_df['movieId'].unique()
            
        # Calculate predicted ratings for all unrated movies
        predicted_ratings = []
        
        for movie_id in all_movies:
            if movie_id not in rated_movies:
                try:
                    predicted_ratings.append((movie_id, self.predict(user_id, movie_id)))
                except Exception as e:
                    print(f"Error predicting rating for movie {movie_id}: {e}")
        
        # Sort by predicted rating
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top N recommendations
        top_recommendations = predicted_ratings[:n_recommendations]
        
        # Get the movie details if available
        if self.movies_df is not None:
            recommended_movies = self.movies_df[self.movies_df['movieId'].isin([x[0] for x in top_recommendations])]
            # Add predicted ratings
            recommended_movies = recommended_movies.copy()
            recommended_movies['predicted_rating'] = [x[1] for x in top_recommendations]
            return recommended_movies
        else:
            return pd.DataFrame({
                'movieId': [x[0] for x in top_recommendations],
                'predicted_rating': [x[1] for x in top_recommendations]
            })
            
    @classmethod
    def create_default_hybrid(cls, ratings_df=None, movies_df=None):
        """
        Create a default hybrid recommender with standard weights.
        
        Args:
            ratings_df (DataFrame, optional): DataFrame containing user ratings
            movies_df (DataFrame, optional): DataFrame containing movie details
            
        Returns:
            HybridRecommender: Configured hybrid recommender
        """
        hybrid = cls(name="DefaultHybrid")
        
        # Add recommenders with weights
        hybrid.add_recommender(ContentBasedRecommender(), weight=0.3)
        hybrid.add_recommender(UserBasedCF(), weight=0.2)
        hybrid.add_recommender(ItemBasedCF(), weight=0.2)
        hybrid.add_recommender(SVDRecommender(), weight=0.3)
        
        # Train if data is provided
        if ratings_df is not None:
            hybrid.fit(ratings_df, movies_df)
            
        return hybrid 