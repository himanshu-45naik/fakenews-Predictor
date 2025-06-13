# Fake News Detection Web Application

A machine learning-powered web application that detects fake news using multiple classification algorithms. The application is built with Flask and uses various ML models including Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest classifiers.

## Features

- **Multi-Model Prediction**: Uses 4 different machine learning algorithms for comprehensive analysis
- **Real-time Detection**: Instant fake news detection through a web interface
- **Text Preprocessing**: Advanced NLP preprocessing including text cleaning and TF-IDF vectorization
- **User-friendly Interface**: Clean and responsive web interface for easy interaction

## Machine Learning Models

The application employs four different classifiers:

1. **Logistic Regression (LR)**: Linear classification algorithm
2. **Decision Tree Classifier (DT)**: Tree-based classification
3. **Gradient Boosting Classifier (GBC)**: Ensemble boosting method
4. **Random Forest Classifier (RFC)**: Ensemble bagging method

## Dataset Requirements

The application requires two CSV files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains legitimate news articles

Both datasets should have the following columns:
- `title`: Article title
- `text`: Article content
- `subject`: News category
- `date`: Publication date

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

Install the required packages using pip:

```bash
pip install flask pandas numpy matplotlib nltk scikit-learn
```

### NLTK Data

Download required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Project Structure

```
fake-news-detection/
│
├── app.py                 # Flask application
├── fakenews.py           # ML model training and prediction
├── Fake.csv              # Fake news dataset
├── True.csv              # Legitimate news dataset
├── static/
│   └── style.css         # CSS styling
├── templates/
│   └── index.html        # HTML template
└── README.md             # Project documentation
```

## Usage

### Running the Application

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fake-news-detection
   ```

2. **Ensure datasets are in place**:
   - Place `Fake.csv` and `True.csv` in the root directory

3. **Run the Flask application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   - Open your web browser
   - Navigate to `http://localhost:5000`

### Using the Interface

1. Enter or paste the news article text in the textarea
2. Click the "Check" button
3. View the predictions from all four models

## How It Works

### Data Preprocessing

1. **Data Loading**: Loads fake and true news datasets
2. **Text Cleaning**: Removes URLs, HTML tags, punctuation, and special characters
3. **Labeling**: Assigns labels (0 for fake, 1 for real)
4. **Feature Extraction**: Uses TF-IDF vectorization to convert text to numerical features

### Model Training

The application trains four different classifiers on the preprocessed data:
- Each model learns from the same training set
- Models are evaluated on a test set (25% of data)
- All models are saved in memory for real-time predictions

### Prediction Process

1. User input is preprocessed using the same cleaning pipeline
2. Text is transformed using the fitted TF-IDF vectorizer
3. All four models make predictions on the processed text
4. Results are displayed showing each model's prediction

## Text Preprocessing Steps

The `wordopt()` function performs:
- Converts text to lowercase
- Removes square brackets and content
- Removes parentheses (including Reuters mentions)
- Removes URLs and HTML tags
- Removes punctuation and special characters
- Removes numbers and newlines

## Model Performance

Each model provides binary classification:
- **0**: Fake News
- **1**: Not Fake News 

The application displays predictions from all models, allowing users to see consensus or disagreement between different algorithms.

## Technical Details

### Backend (`fakenews.py`)
- Data preprocessing and model training
- TF-IDF vectorization for feature extraction
- Multiple classifier implementation
- Prediction function for new text input

### Frontend (`app.py`)
- Flask web application setup
- Route handling for GET/POST requests
- Integration with ML backend

### Styling (`static/style.css`)
- Responsive design
- Clean and professional appearance
- Form styling and result display
