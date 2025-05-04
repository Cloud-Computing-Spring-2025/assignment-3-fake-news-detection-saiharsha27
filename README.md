# Assignment-5-FakeNews-Detection

This assignment implements a fake news detection system using Apache Spark's MLlib. The system analyzes text content and classifies news articles as either FAKE or REAL using a machine learning approach.

## Dataset

The dataset used in this assignment is `fake_news_sample.csv`, which contains news articles with the following fields:
- `id`: Unique identifier for each article
- `title`: The headline of the article
- `text`: The content/body of the article
- `label`: Classification as either "FAKE" or "REAL"

The dataset was generated using the code provided in the GitHub repository. It contains a balanced mixture of fake and real news articles with corresponding features.

## Assignment Tasks

The assignment is divided into 5 separate tasks, each implemented in its own Python file:

### Task 1: Load & Basic Exploration (`task1.py`)
- Loads the CSV data into a Spark DataFrame
- Creates a temporary view for SQL queries
- Performs basic analysis (counts, distinct values)
- Shows first few rows of the dataset

### Task 2: Text Preprocessing (`task2.py`)
- Converts text to lowercase
- Tokenizes text into individual words
- Removes stopwords (common, non-informative words)
- Outputs the cleaned text data

### Task 3: Feature Extraction (`task3.py`)
- Creates TF-IDF vectors from the tokenized text
- Converts string labels to numeric indices
- Prepares the data for machine learning algorithms

### Task 4: Model Training (`task4.py`)
- Splits data into training (80%) and test (20%) sets
- Trains a Logistic Regression model
- Makes predictions on the test set

### Task 5: Model Evaluation (`task5.py`)
- Evaluates the model using accuracy and F1 score
- Creates a confusion matrix
- Outputs evaluation metrics

## How to Run the Code

### Prerequisites
- Apache Spark (PySpark)
- Python 3.x
- The `fake_news_sample.csv` file in the assignment directory

### Execution Steps

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Make sure the dataset is in the correct location:
   ```
   # The code expects to find fake_news_sample.csv in the current directory
   ```

3. Run each task in sequence:
   ```
   python task1.py  # Load & explore data
   python task2.py  # Preprocess text
   python task3.py  # Extract features
   python task4.py  # Train model
   python task5.py  # Evaluate model
   ```

4. Output files:
   - Results for each task will be saved in the `output` directory
   - Each task creates its own subdirectory (e.g., `output/task1_output`, `output/task2_output`, etc.)