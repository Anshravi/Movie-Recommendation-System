# Movie Recommendation System

A personalized movie recommendation system that suggests movies based on user preferences, ratings, and viewing history. The system uses collaborative filtering, content-based filtering, and a hybrid approach to provide accurate and diverse recommendations.

## Features

- Personalized recommendations based on past ratings
- Option to filter movies by genre, release year, or popularity
- User input for real-time recommendations
- Scalable model to handle new data
- Web interface for easy interaction
- API endpoints for integration with other applications

## Project Structure

```
Movie Recommendation System/
├── data/                  # Dataset storage
├── models/                # Trained recommendation models
├── plots/                 # Evaluation plots
├── src/                   # Source code
│   ├── models/            # Recommendation models
│   │   ├── base_model.py  # Base recommender class
│   │   ├── content_based.py # Content-based filtering
│   │   ├── collaborative_filtering.py # Collaborative filtering
│   │   └── hybrid.py      # Hybrid recommender
│   ├── utils/             # Utility functions
│   │   ├── data_loader.py # Data loading and preprocessing
│   │   ├── evaluation.py  # Model evaluation metrics
│   │   └── download_data.py # Dataset download script
│   ├── demo.py            # Console demo script
│   └── train_models.py    # Model training script
├── static/                # Static files for web app
│   ├── css/               # CSS stylesheets
│   └── js/                # JavaScript files
├── templates/             # HTML templates
│   ├── index.html         # Home page
│   ├── recommendations.html # Recommendations page
│   └── movie_details.html # Movie details page
├── app.py                 # Flask web application
├── run.py                 # Script to run the application
├── setup.py               # Setup script
└── requirements.txt       # Dependencies
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Run the setup script to install dependencies and download the dataset:
   ```
   python setup.py
   ```

3. Train the recommendation models:
   ```
   python run.py --train
   ```

4. Run the web application:
   ```
   python run.py --run
   ```

5. Alternatively, you can run all steps at once:
   ```
   python run.py --all
   ```

## Usage

### Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Enter a user ID (1-610 for the MovieLens dataset)
3. Set the number of recommendations you want to receive
4. Optionally, filter by genre, release year, or other criteria
5. Click "Get Recommendations" to see personalized movie suggestions

### Console Demo

Run the console demo to see recommendations from different models:
```
python -m src.demo
```

### API Endpoints

The system provides the following API endpoints:

- `GET /api/recommend/<user_id>?n=10`: Get recommendations for a user
  - `user_id`: User ID
  - `n`: Number of recommendations (default: 10)

## Approaches Used

### Content-Based Filtering

Recommends movies based on similarities in genres, actors, or descriptions. This approach analyzes the content of movies and recommends similar items based on a user's past preferences.

### Collaborative Filtering

- **User-Based**: Recommends movies that similar users have liked
- **Item-Based**: Recommends movies that are similar to the ones the user has liked
- **Matrix Factorization (SVD)**: Uses singular value decomposition to identify latent factors

### Hybrid Model

Combines content-based and collaborative filtering approaches to provide more accurate and diverse recommendations.

## Evaluation Metrics

The system evaluates recommendation quality using:

- **RMSE (Root Mean Square Error)**: Measures the difference between predicted and actual ratings
- **MAE (Mean Absolute Error)**: Measures the average magnitude of errors
- **Precision@K**: Measures the proportion of recommended items that are relevant

## Future Enhancements

- Implement deep learning models for better recommendations
- Add sentiment analysis on movie reviews
- Integrate with external APIs for real-time movie data
- Implement user authentication and personalized profiles
- Add more advanced filtering options

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for providing the dataset
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms 