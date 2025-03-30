import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from .base_model import BaseRecommender

class UserBasedCF(BaseRecommender):
    """
    User-based collaborative filtering recommender system.
    
    This model recommends items that similar users have liked.
    """
    
    def __init__(self, name="UserBasedCF", min_ratings=5):
        """
        Initialize the user-based collaborative filtering recommender.
        
        Args:
            name (str): Name of the recommender
            min_ratings (int): Minimum number of ratings for a user to be considered
        """
        super().__init__(name)
        self.min_ratings = min_ratings
        self.user_item_matrix = None
        self.user_similarity = None
        self.movies_df = None
        self.ratings_df = None
        self.mean_ratings = None
        
    def fit(self, ratings_df, movies_df=None):
        """
        Train the user-based collaborative filtering model.
        
        Args:
            ratings_df (DataFrame): DataFrame containing user ratings
            movies_df (DataFrame, optional): DataFrame containing movie details
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Calculate mean ratings for each user
        self.mean_ratings = self.user_item_matrix.mean(axis=1)
        
        # Normalize ratings by subtracting mean rating for each user
        user_item_matrix_normalized = self.user_item_matrix.sub(self.mean_ratings, axis=0)
        
        # Calculate user similarity matrix
        self.user_similarity = cosine_similarity(user_item_matrix_normalized)
        
        # Create a DataFrame for the similarity matrix
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        self.is_fitted = True
        
        return self
    
    def _get_similar_users(self, user_id, n=10):
        """
        Get the most similar users to a given user.
        
        Args:
            user_id: User identifier
            n (int): Number of similar users to return
            
        Returns:
            Series: Series of similar users with similarity scores
        """
        if user_id not in self.user_similarity_df.index:
            return pd.Series()
            
        # Get similarity scores for the user
        user_similarities = self.user_similarity_df.loc[user_id]
        
        # Sort by similarity and exclude the user itself
        similar_users = user_similarities.drop(user_id).sort_values(ascending=False)
        
        # Return the top N similar users
        return similar_users.head(n)
    
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
            
        # Check if the user and movie exist in the data
        if user_id not in self.user_item_matrix.index:
            return self.ratings_df['rating'].mean()
            
        if movie_id not in self.user_item_matrix.columns:
            return self.mean_ratings[user_id] if user_id in self.mean_ratings else self.ratings_df['rating'].mean()
            
        # Check if the user has already rated the movie
        if self.user_item_matrix.loc[user_id, movie_id] > 0:
            return self.user_item_matrix.loc[user_id, movie_id]
            
        # Get similar users
        similar_users = self._get_similar_users(user_id)
        
        if len(similar_users) == 0:
            return self.mean_ratings[user_id] if user_id in self.mean_ratings else self.ratings_df['rating'].mean()
            
        # Get ratings for the movie from similar users
        similar_users_ratings = []
        similar_users_similarities = []
        
        for sim_user, similarity in similar_users.items():
            rating = self.user_item_matrix.loc[sim_user, movie_id]
            if rating > 0:  # Only consider users who have rated the movie
                similar_users_ratings.append(rating)
                similar_users_similarities.append(similarity)
                
        # If no similar users have rated the movie, return the user's mean rating
        if len(similar_users_ratings) == 0:
            return self.mean_ratings[user_id] if user_id in self.mean_ratings else self.ratings_df['rating'].mean()
            
        # Calculate the weighted average rating
        weighted_rating = np.sum(np.array(similar_users_ratings) * np.array(similar_users_similarities)) / np.sum(similar_users_similarities)
        
        # Add the user's mean rating to get the final prediction
        predicted_rating = self.mean_ratings[user_id] + weighted_rating - np.mean(similar_users_ratings)
        
        # Clip the rating to be within the valid range
        return max(0.5, min(5.0, predicted_rating))
    
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
            
        # Check if the user exists in the data
        if user_id not in self.user_item_matrix.index:
            # If the user doesn't exist, recommend the most popular movies
            popular_movies = self.ratings_df.groupby('movieId')['rating'].count().sort_values(ascending=False)
            if self.movies_df is not None:
                recommendations = self.movies_df[self.movies_df['movieId'].isin(popular_movies.index[:n_recommendations])]
                return recommendations
            else:
                return pd.DataFrame({'movieId': popular_movies.index[:n_recommendations]})
        
        # Get the movies the user has already rated
        rated_movies = self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index if exclude_rated else []
        
        # Get all movies
        all_movies = self.user_item_matrix.columns
        
        # Calculate predicted ratings for all unrated movies
        predicted_ratings = []
        
        for movie_id in all_movies:
            if movie_id not in rated_movies:
                predicted_ratings.append((movie_id, self.predict(user_id, movie_id)))
        
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


class ItemBasedCF(BaseRecommender):
    """
    Item-based collaborative filtering recommender system.
    
    This model recommends items that are similar to the items that a user has liked in the past.
    """
    
    def __init__(self, name="ItemBasedCF"):
        """
        Initialize the item-based collaborative filtering recommender.
        
        Args:
            name (str): Name of the recommender
        """
        super().__init__(name)
        self.user_item_matrix = None
        self.item_similarity = None
        self.movies_df = None
        self.ratings_df = None
        
    def fit(self, ratings_df, movies_df=None):
        """
        Train the item-based collaborative filtering model.
        
        Args:
            ratings_df (DataFrame): DataFrame containing user ratings
            movies_df (DataFrame, optional): DataFrame containing movie details
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Calculate item similarity matrix
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        # Create a DataFrame for the similarity matrix
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        self.is_fitted = True
        
        return self
    
    def _get_similar_items(self, item_id, n=10):
        """
        Get the most similar items to a given item.
        
        Args:
            item_id: Item identifier
            n (int): Number of similar items to return
            
        Returns:
            Series: Series of similar items with similarity scores
        """
        if item_id not in self.item_similarity_df.index:
            return pd.Series()
            
        # Get similarity scores for the item
        item_similarities = self.item_similarity_df.loc[item_id]
        
        # Sort by similarity and exclude the item itself
        similar_items = item_similarities.drop(item_id).sort_values(ascending=False)
        
        # Return the top N similar items
        return similar_items.head(n)
    
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
            
        # Check if the user and movie exist in the data
        if user_id not in self.user_item_matrix.index:
            return self.ratings_df['rating'].mean()
            
        if movie_id not in self.item_similarity_df.index:
            return self.ratings_df['rating'].mean()
            
        # Check if the user has already rated the movie
        if self.user_item_matrix.loc[user_id, movie_id] > 0:
            return self.user_item_matrix.loc[user_id, movie_id]
            
        # Get the user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        if len(rated_items) == 0:
            return self.ratings_df['rating'].mean()
            
        # Get similar items to the target movie
        similar_items = self._get_similar_items(movie_id)
        
        if len(similar_items) == 0:
            return self.ratings_df['rating'].mean()
            
        # Calculate the weighted average rating
        weighted_sum = 0
        similarity_sum = 0
        
        for item_id in rated_items:
            if item_id in similar_items:
                similarity = similar_items[item_id]
                rating = user_ratings[item_id]
                weighted_sum += similarity * rating
                similarity_sum += similarity
                
        # If no similar items have been rated by the user, return the average rating
        if similarity_sum == 0:
            return self.ratings_df['rating'].mean()
            
        predicted_rating = weighted_sum / similarity_sum
        
        # Clip the rating to be within the valid range
        return max(0.5, min(5.0, predicted_rating))
    
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
            
        # Check if the user exists in the data
        if user_id not in self.user_item_matrix.index:
            # If the user doesn't exist, recommend the most popular movies
            popular_movies = self.ratings_df.groupby('movieId')['rating'].count().sort_values(ascending=False)
            if self.movies_df is not None:
                recommendations = self.movies_df[self.movies_df['movieId'].isin(popular_movies.index[:n_recommendations])]
                return recommendations
            else:
                return pd.DataFrame({'movieId': popular_movies.index[:n_recommendations]})
        
        # Get the movies the user has already rated
        rated_movies = self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index if exclude_rated else []
        
        # Get all movies
        all_movies = self.item_similarity_df.index
        
        # Calculate predicted ratings for all unrated movies
        predicted_ratings = []
        
        for movie_id in all_movies:
            if movie_id not in rated_movies:
                predicted_ratings.append((movie_id, self.predict(user_id, movie_id)))
        
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


class SVDRecommender(BaseRecommender):
    """
    Matrix factorization recommender using Singular Value Decomposition (SVD).
    """
    
    def __init__(self, name="SVD", n_factors=50):
        """
        Initialize the SVD recommender.
        
        Args:
            name (str): Name of the recommender
            n_factors (int): Number of latent factors
        """
        super().__init__(name)
        self.n_factors = n_factors
        self.user_item_matrix = None
        self.user_features = None
        self.item_features = None
        self.mean_rating = None
        self.movies_df = None
        self.ratings_df = None
        self.user_ids = None
        self.movie_ids = None
        
    def fit(self, ratings_df, movies_df=None):
        """
        Train the SVD recommendation model.
        
        Args:
            ratings_df (DataFrame): DataFrame containing user ratings
            movies_df (DataFrame, optional): DataFrame containing movie details
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Store user and movie IDs
        self.user_ids = self.user_item_matrix.index.tolist()
        self.movie_ids = self.user_item_matrix.columns.tolist()
        
        # Convert to numpy array
        ratings_matrix = self.user_item_matrix.values
        
        # Calculate the mean rating
        self.mean_rating = np.mean(ratings_matrix[ratings_matrix > 0])
        
        # Normalize the ratings by subtracting the mean
        normalized_matrix = ratings_matrix.copy()
        normalized_matrix[normalized_matrix > 0] -= self.mean_rating
        
        # Perform SVD
        u, sigma, vt = svds(normalized_matrix, k=min(self.n_factors, min(normalized_matrix.shape) - 1))
        
        # Convert sigma to a diagonal matrix
        sigma_diag = np.diag(sigma)
        
        # Store the user and item features
        self.user_features = u
        self.item_features = vt.T
        
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
            
        # Check if the user and movie exist in the data
        if user_id not in self.user_ids or movie_id not in self.movie_ids:
            return self.mean_rating
            
        # Get the indices
        user_idx = self.user_ids.index(user_id)
        movie_idx = self.movie_ids.index(movie_id)
        
        # Calculate the predicted rating
        predicted_rating = self.mean_rating + np.dot(self.user_features[user_idx, :], self.item_features[movie_idx, :])
        
        # Clip the rating to be within the valid range
        return max(0.5, min(5.0, predicted_rating))
    
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
            
        # Check if the user exists in the data
        if user_id not in self.user_ids:
            # If the user doesn't exist, recommend the most popular movies
            popular_movies = self.ratings_df.groupby('movieId')['rating'].count().sort_values(ascending=False)
            if self.movies_df is not None:
                recommendations = self.movies_df[self.movies_df['movieId'].isin(popular_movies.index[:n_recommendations])]
                return recommendations
            else:
                return pd.DataFrame({'movieId': popular_movies.index[:n_recommendations]})
        
        # Get the user's index
        user_idx = self.user_ids.index(user_id)
        
        # Get the movies the user has already rated
        if exclude_rated:
            user_ratings = self.user_item_matrix.loc[user_id]
            rated_movies = user_ratings[user_ratings > 0].index.tolist()
        else:
            rated_movies = []
        
        # Calculate predicted ratings for all movies
        predicted_ratings = []
        
        for movie_id in self.movie_ids:
            if movie_id not in rated_movies:
                predicted_ratings.append((movie_id, self.predict(user_id, movie_id)))
        
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