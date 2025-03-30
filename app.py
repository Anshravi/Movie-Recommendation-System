import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from src.utils.data_loader import DataLoader
from src.models.hybrid import HybridRecommender

app = Flask(__name__)

# Load data and models
@app.before_first_request
def load_data_and_models():
    global ratings_df, movies_df, model, movie_genres
    
    # Load data
    data_loader = DataLoader()
    ratings_df, movies_df, _ = data_loader.load_movielens_data()
    ratings_df, movies_df, _ = data_loader.preprocess_data()
    
    # Extract unique genres
    all_genres = []
    for genres in movies_df['genres']:
        if isinstance(genres, list):
            all_genres.extend(genres)
        elif isinstance(genres, str):
            all_genres.extend(genres.split('|'))
    movie_genres = sorted(list(set(all_genres)))
    
    # Load the hybrid model
    model_path = os.path.join('models', 'Hybrid.joblib')
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        # If the model doesn't exist, create a new one
        from src.train_models import train_and_evaluate_models
        models = train_and_evaluate_models()
        model = models['Hybrid']

@app.route('/')
def index():
    # Get the top 10 most popular movies
    popular_movies = ratings_df.groupby('movieId')['rating'].count().sort_values(ascending=False).head(10)
    popular_movies_data = movies_df[movies_df['movieId'].isin(popular_movies.index)]
    
    # Convert genres from list to string for display
    popular_movies_data = popular_movies_data.copy()
    popular_movies_data['genres_str'] = popular_movies_data['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    return render_template('index.html', 
                          popular_movies=popular_movies_data.to_dict('records'),
                          genres=movie_genres)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input
    user_id = int(request.form.get('userId', 1))
    n_recommendations = int(request.form.get('n_recommendations', 10))
    selected_genres = request.form.getlist('genres')
    min_year = request.form.get('min_year', '')
    max_year = request.form.get('max_year', '')
    
    # Generate recommendations
    recommendations = model.recommend(user_id, n_recommendations=n_recommendations*2, exclude_rated=True)
    
    # Filter by genre if selected
    if selected_genres:
        filtered_recommendations = []
        for _, movie in recommendations.iterrows():
            movie_genres = movie['genres']
            if any(genre in movie_genres for genre in selected_genres):
                filtered_recommendations.append(movie)
        recommendations = pd.DataFrame(filtered_recommendations)
    
    # Filter by year if provided
    if min_year and min_year.isdigit():
        recommendations = recommendations[recommendations['year'].astype(int) >= int(min_year)]
    if max_year and max_year.isdigit():
        recommendations = recommendations[recommendations['year'].astype(int) <= int(max_year)]
    
    # Take the top N recommendations
    recommendations = recommendations.head(n_recommendations)
    
    # Convert genres from list to string for display
    recommendations = recommendations.copy()
    recommendations['genres_str'] = recommendations['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    return render_template('recommendations.html', 
                          recommendations=recommendations.to_dict('records'),
                          user_id=user_id)

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    # Get movie details
    movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
    
    # Get similar movies
    from src.models.content_based import ContentBasedRecommender
    content_model_path = os.path.join('models', 'ContentBased.joblib')
    
    if os.path.exists(content_model_path):
        content_model = joblib.load(content_model_path)
    else:
        content_model = ContentBasedRecommender()
        content_model.fit(ratings_df, movies_df)
    
    # Get similar movies based on content
    movie_idx = content_model.movie_indices[movie_id]
    sim_scores = list(enumerate(content_model.cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar movies, excluding itself
    movie_indices = [i[0] for i in sim_scores]
    similar_movies = movies_df.iloc[movie_indices]
    
    # Convert genres from list to string for display
    movie = movie.copy()
    movie['genres_str'] = ', '.join(movie['genres']) if isinstance(movie['genres'], list) else movie['genres']
    
    similar_movies = similar_movies.copy()
    similar_movies['genres_str'] = similar_movies['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    # Get ratings for this movie
    movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]
    avg_rating = movie_ratings['rating'].mean() if len(movie_ratings) > 0 else 0
    num_ratings = len(movie_ratings)
    
    return render_template('movie_details.html', 
                          movie=movie,
                          similar_movies=similar_movies.to_dict('records'),
                          avg_rating=avg_rating,
                          num_ratings=num_ratings)

@app.route('/api/recommend/<int:user_id>')
def api_recommend(user_id):
    n_recommendations = int(request.args.get('n', 10))
    
    # Generate recommendations
    recommendations = model.recommend(user_id, n_recommendations=n_recommendations, exclude_rated=True)
    
    # Convert to JSON
    result = []
    for _, movie in recommendations.iterrows():
        result.append({
            'movieId': int(movie['movieId']),
            'title': movie['title'],
            'genres': movie['genres'] if isinstance(movie['genres'], list) else movie['genres'].split('|'),
            'year': movie['year'] if 'year' in movie else None,
            'predicted_rating': float(movie['predicted_rating']) if 'predicted_rating' in movie else None
        })
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 