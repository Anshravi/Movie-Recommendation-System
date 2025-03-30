// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize any components that need JavaScript
    initializeRatingForm();
    initializeYearFilters();
});

// Initialize the rating form
function initializeRatingForm() {
    const ratingForm = document.getElementById('rating-form');
    if (ratingForm) {
        ratingForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const userId = document.getElementById('userId').value;
            const rating = document.getElementById('rating').value;
            const movieId = document.querySelector('[data-movie-id]')?.dataset.movieId;
            
            // In a real application, you would send this data to the server
            // For now, just show a success message
            const messageElement = document.getElementById('rating-message');
            if (messageElement) {
                messageElement.innerHTML = '<div class="alert alert-success">Thank you for rating this movie!</div>';
                
                // Clear the message after 3 seconds
                setTimeout(function() {
                    messageElement.innerHTML = '';
                }, 3000);
            }
        });
    }
}

// Initialize year filters
function initializeYearFilters() {
    const minYearInput = document.getElementById('min_year');
    const maxYearInput = document.getElementById('max_year');
    
    if (minYearInput && maxYearInput) {
        // Set default values if not already set
        if (!minYearInput.value) {
            minYearInput.value = '1900';
        }
        
        if (!maxYearInput.value) {
            const currentYear = new Date().getFullYear();
            maxYearInput.value = currentYear.toString();
        }
        
        // Ensure min year is not greater than max year
        minYearInput.addEventListener('change', function() {
            if (parseInt(minYearInput.value) > parseInt(maxYearInput.value)) {
                maxYearInput.value = minYearInput.value;
            }
        });
        
        // Ensure max year is not less than min year
        maxYearInput.addEventListener('change', function() {
            if (parseInt(maxYearInput.value) < parseInt(minYearInput.value)) {
                minYearInput.value = maxYearInput.value;
            }
        });
    }
}

// Function to toggle genre selection
function toggleAllGenres(select) {
    const checkboxes = document.querySelectorAll('input[name="genres"]');
    checkboxes.forEach(checkbox => {
        checkbox.checked = select;
    });
}

// Function to get recommendations via API
function getRecommendationsApi(userId, count = 10) {
    fetch(`/api/recommend/${userId}?n=${count}`)
        .then(response => response.json())
        .then(data => {
            console.log('Recommendations:', data);
            // In a real application, you would display these recommendations
        })
        .catch(error => {
            console.error('Error fetching recommendations:', error);
        });
} 