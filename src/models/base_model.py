from abc import ABC, abstractmethod
import joblib
import os

class BaseRecommender(ABC):
    """
    Abstract base class for recommendation models.
    """
    
    def __init__(self, name):
        """
        Initialize the base recommender.
        
        Args:
            name (str): Name of the recommender
        """
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, train_data):
        """
        Train the recommendation model.
        
        Args:
            train_data: Training data
        """
        pass
    
    @abstractmethod
    def predict(self, user_id, item_id):
        """
        Predict the rating for a user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            float: Predicted rating
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations (int): Number of recommendations to generate
            exclude_rated (bool): Whether to exclude items the user has already rated
            
        Returns:
            list: List of recommended item IDs
        """
        pass
    
    def save_model(self, model_dir='models'):
        """
        Save the model to disk.
        
        Args:
            model_dir (str): Directory to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(model_dir, f"{self.name}.joblib")
        joblib.dump(self, model_path)
        
        return model_path
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load a model from disk.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            BaseRecommender: Loaded model
        """
        return joblib.load(model_path) 