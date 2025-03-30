import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Utility class for loading and preprocessing movie recommendation datasets.
    """
    
    def __init__(self, data_path='data'):
        """
        Initialize the DataLoader with the path to the data directory.
        
        Args:
            data_path (str): Path to the directory containing the datasets
        """
        self.data_path = data_path
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        
    def load_movielens_data(self, size='small'):
        """
        Load the MovieLens dataset.
        
        Args:
            size (str): Size of the dataset ('small', 'full')
            
        Returns:
            tuple: (ratings_df, movies_df, users_df)
        """
        if size == 'small':
            ratings_file = os.path.join(self.data_path, 'ratings.csv')
            movies_file = os.path.join(self.data_path, 'movies.csv')
            
            # Check if files exist
            if not (os.path.exists(ratings_file) and os.path.exists(movies_file)):
                raise FileNotFoundError(f"MovieLens dataset files not found in {self.data_path}. "
                                       f"Please download the dataset and place it in the data directory.")
            
            # Load the data
            self.ratings_df = pd.read_csv(ratings_file)
            self.movies_df = pd.read_csv(movies_file)
            
            # Try to load users data if available
            users_file = os.path.join(self.data_path, 'users.csv')
            if os.path.exists(users_file):
                self.users_df = pd.read_csv(users_file)
        
        return self.ratings_df, self.movies_df, self.users_df
    
    def preprocess_data(self):
        """
        Preprocess the loaded data.
        
        Returns:
            tuple: (processed_ratings_df, processed_movies_df, processed_users_df)
        """
        if self.ratings_df is None or self.movies_df is None:
            raise ValueError("Data not loaded. Call load_movielens_data() first.")
        
        # Handle missing values
        self.ratings_df = self.ratings_df.dropna()
        self.movies_df = self.movies_df.dropna(subset=['title', 'genres'])
        
        # Extract year from title if not already a separate column
        if 'year' not in self.movies_df.columns:
            self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)$')
            
        # Convert genres from string to list
        if isinstance(self.movies_df['genres'].iloc[0], str):
            self.movies_df['genres'] = self.movies_df['genres'].apply(lambda x: x.split('|'))
            
        return self.ratings_df, self.movies_df, self.users_df
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the ratings data into training and testing sets.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_data, test_data)
        """
        if self.ratings_df is None:
            raise ValueError("Data not loaded. Call load_movielens_data() first.")
            
        train_data, test_data = train_test_split(
            self.ratings_df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        return train_data, test_data
    
    def create_user_item_matrix(self):
        """
        Create a user-item matrix from the ratings data.
        
        Returns:
            DataFrame: User-item matrix where rows are users, columns are movies, and values are ratings
        """
        if self.ratings_df is None:
            raise ValueError("Data not loaded. Call load_movielens_data() first.")
            
        user_item_matrix = self.ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        return user_item_matrix 