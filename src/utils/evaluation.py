import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error.
    
    Args:
        y_true (array-like): True ratings
        y_pred (array-like): Predicted ratings
        
    Returns:
        float: RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true (array-like): True ratings
        y_pred (array-like): Predicted ratings
        
    Returns:
        float: MAE value
    """
    return mean_absolute_error(y_true, y_pred)

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate Precision@K.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of relevant (true positive) item IDs
        k (int): Number of recommendations to consider
        
    Returns:
        float: Precision@K value
    """
    # Take only the top-k recommendations
    recommended_items = recommended_items[:k]
    
    # Count the number of relevant items in the recommendations
    num_relevant = len(set(recommended_items) & set(relevant_items))
    
    # Calculate precision
    precision = num_relevant / min(k, len(recommended_items)) if len(recommended_items) > 0 else 0
    
    return precision

def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate Recall@K.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of relevant (true positive) item IDs
        k (int): Number of recommendations to consider
        
    Returns:
        float: Recall@K value
    """
    # Take only the top-k recommendations
    recommended_items = recommended_items[:k]
    
    # Count the number of relevant items in the recommendations
    num_relevant = len(set(recommended_items) & set(relevant_items))
    
    # Calculate recall
    recall = num_relevant / len(relevant_items) if len(relevant_items) > 0 else 0
    
    return recall

def evaluate_recommendations(model, test_data, k=10):
    """
    Evaluate a recommendation model using various metrics.
    
    Args:
        model: Recommendation model with a predict method
        test_data (DataFrame): Test data containing user-item interactions
        k (int): Number of recommendations to consider for precision and recall
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Extract true ratings
    true_ratings = test_data['rating'].values
    
    # Generate predictions
    user_ids = test_data['userId'].values
    movie_ids = test_data['movieId'].values
    predicted_ratings = []
    
    for user_id, movie_id in zip(user_ids, movie_ids):
        try:
            pred = model.predict(user_id, movie_id)
            predicted_ratings.append(pred)
        except:
            # If prediction fails, use the mean rating
            predicted_ratings.append(test_data['rating'].mean())
    
    # Calculate error metrics
    rmse = calculate_rmse(true_ratings, predicted_ratings)
    mae = calculate_mae(true_ratings, predicted_ratings)
    
    # Return metrics
    return {
        'RMSE': rmse,
        'MAE': mae
    } 