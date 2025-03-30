import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseRecommender

class ContentBasedRecommender(BaseRecommender):
    """
    Content-based filtering recommender system.
    
    This model recommends items that are similar to the items that a user has liked in the past.
    It uses movie features such as genres, actors, directors, etc. to calculate similarity.
    """
    
    def __init__(self, name="ContentBased"):
        """
        Initialize the content-based recommender.
        
        Args:
            name (str): Name of the recommender
        """
        super().__init__(name)
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.movie_indices = None
        
    def _create_feature_soup(self, row):
        """
        Create a 'soup' of features for a movie.
        
        Args:
            row (Series): Row from the movies DataFrame
            
        Returns:
            str: String containing all features
        """
        features = []
        
        # Add genres
        if isinstance(row['genres'], list):
            features.extend(row['genres'])
        elif isinstance(row['genres'], str):
            features.extend(row['genres'].split('|'))
            
        # Add other features if available
        for feature in ['actors', 'director', 'keywords']:
            if feature in row and pd.notna(row[feature]):
                if isinstance(row[feature], list):
                    features.extend(row[feature])
                elif isinstance(row[feature], str):
                    features.extend(row[feature].split('|'))
        
        # Return the soup
        return ' '.join(features)
    
    def fit(self, ratings_df, movies_df):
        """
        Train the content-based recommendation model.
        
        Args:
            ratings_df (DataFrame): DataFrame containing user ratings
            movies_df (DataFrame): DataFrame containing movie features
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Create a feature soup for each movie
        self.movies_df['soup'] = self.movies_df.apply(self._create_feature_soup, axis=1)
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['soup'])
        
        # Calculate cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create a mapping from movie ID to index
        self.movie_indices = pd.Series(
            self.movies_df.index, 
            index=self.movies_df['movieId']
        ).drop_duplicates()
        
        self.is_fitted = True
        
        return self
    
    def _get_movie_similarity(self, movie_id):
        """
        Get the similarity scores for a movie.
        
        Args:
            movie_id: Movie identifier
            
        Returns:
            array: Array of similarity scores
        """
        if movie_id not in self.movie_indices:
            return np.zeros(len(self.movies_df))
            
        idx = self.movie_indices[movie_id]
        return self.cosine_sim[idx]
    
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
            
        # Get the user's ratings
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            # If the user has no ratings, return the average rating
            return self.ratings_df['rating'].mean()
        
        # Get similarity scores for the target movie
        sim_scores = self._get_movie_similarity(movie_id)
        
        # Get the indices of the movies the user has rated
        rated_indices = [self.movie_indices[mid] for mid in user_ratings['movieId'] 
                         if mid in self.movie_indices]
        
        # Get the similarity scores for the rated movies
        rated_sims = sim_scores[rated_indices]
        
        # Get the user's ratings for the rated movies
        user_ratings_values = user_ratings['rating'].values
        
        # Calculate the weighted average rating
        if len(rated_sims) > 0 and np.sum(rated_sims) > 0:
            weighted_rating = np.sum(rated_sims * user_ratings_values) / np.sum(rated_sims)
            return weighted_rating
        else:
            # If no similar movies, return the average rating
            return self.ratings_df['rating'].mean()
    
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
            
        # Get the user's ratings
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            # If the user has no ratings, recommend the most popular movies
            popular_movies = self.ratings_df.groupby('movieId')['rating'].count().sort_values(ascending=False)
            recommendations = self.movies_df[self.movies_df['movieId'].isin(popular_movies.index[:n_recommendations])]
            return recommendations
        
        # Get the movies the user has already rated
        rated_movies = user_ratings['movieId'].values if exclude_rated else []
        
        # Calculate the average similarity to the user's rated movies for each movie
        all_movies = self.movies_df['movieId'].values
        predicted_ratings = []
        
        for movie_id in all_movies:
            if movie_id not in rated_movies:
                predicted_ratings.append((movie_id, self.predict(user_id, movie_id)))
        
        # Sort by predicted rating
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top N recommendations
        top_recommendations = predicted_ratings[:n_recommendations]
        
        # Get the movie details
        recommended_movies = self.movies_df[self.movies_df['movieId'].isin([x[0] for x in top_recommendations])]
        
        return recommended_movies 