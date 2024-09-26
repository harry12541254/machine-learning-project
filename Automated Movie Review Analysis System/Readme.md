# Sentiment Classification Using Logistic Regression

This project implements a simple sentiment classification model using logistic regression. The model takes text input and classifies the sentiment as positive or negative by extracting word and character-based features and learning the relationships using gradient descent.

## Project Structure

- `submission.py`: Contains the core implementation for the sentiment classification model using logistic regression. This includes feature extraction, model training (`learnPredictor`), and dataset generation.
- `polarity.train`: The training dataset for sentiment classification.
- `polarity.dev`: The validation dataset for evaluating model performance.
- `util.py`: A utility file that includes helper functions such as `dotProduct`, `increment`, and others to facilitate vector operations.
- `error-analysis`: Stores the output of the model’s error analysis.
- `weights`: Stores the learned weight vector after training the model.
- `grader.py`, `graderUtil.py`, `grader-all.js`: Files used to evaluate the implementation and ensure correctness.
- `interactive.py`: Used for running the model in an interactive environment for debugging and testing.
- `dictionary.txt`: A dictionary file possibly used in the model’s feature extraction.
- `__pycache__`: Stores compiled Python files.

In the submission.py, the function is define below:

- `extractWordFeatures(x: str)`: Extracts word frequency features from a given string.
- `learnPredictor()`: Implements logistic regression using gradient descent to learn the weight vector based on the training data.
- `generateDataset()`: Generates a dataset with examples classified by the learned weight vector.
- `extractCharacterFeatures(n: int)`: Extracts n-gram character features from a string.
- `testValuesOfN(n: int)`: Tests the model with different values of `n` for n-gram features.

## Features

1. **Word Features**: Extracts features based on word frequency.
2. **Character N-gram Features**: Allows character-level feature extraction for more granular analysis (e.g., capturing sentiment nuances in shorter texts).
3. **Logistic Regression**: Uses sigmoid function for classification and optimizes weights using gradient descent.

## How It Works

1. **Feature Extraction**: 
   - The model supports both word-level and character-level n-gram features. For instance, for a sentence like `"I like tacos"`, the word feature extractor will count occurrences of each word, while the character feature extractor will consider substrings of length `n`.

2. **Logistic Regression**: 
   - The logistic regression model is trained using the gradient descent method. Each feature (either words or character n-grams) is weighted according to its importance in predicting the sentiment.

3. **Training and Evaluation**: 
   - The training process involves minimizing the error between the predicted and actual sentiments using cross-entropy loss.
   - Errors on both training and validation datasets are printed after each epoch to track performance.

## Setup and Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sentiment-classification.git
    ```
   
2. Navigate to the project directory:
    ```bash
    cd sentiment-classification
    ```

3. Install dependencies (if any):
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can train the model with a custom dataset or test with the provided test cases.

### Train the Model
To train the model with word features or character n-gram features, you can modify the `testValuesOfN()` function:

```python
testValuesOfN(3)  # For example, testing with 3-gram character features
