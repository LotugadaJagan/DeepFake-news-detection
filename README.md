# Fake News Detection Web Application

A web application that uses machine learning to detect fake news articles. This application leverages a TensorFlow LSTM (Long Short-Term Memory) neural network trained on the Kaggle fake-and-real-news dataset to analyze and classify news content as either potentially fake or authentic.

## Features

- **Text Analysis**: Input news text and get an instant prediction with confidence score
- **Visual Results**: Color-coded results with confidence percentages
- **Batch Processing**: Upload CSV or TXT files for analyzing multiple articles at once
- **Example Snippets**: Pre-loaded example texts to test the system
- **History Tracking**: View your last 5 predictions
- **Model Information**: Detailed metrics and statistics about the machine learning model
- **Mobile Responsive**: Works on all device sizes

## How It Works

1. **Text Preprocessing**: News content is cleaned, normalized, and tokenized
2. **LSTM Analysis**: The neural network analyzes patterns in the text that are indicative of fake news
3. **Prediction Generation**: The model outputs a prediction (fake/real) with a confidence percentage
4. **Result Visualization**: Results are displayed with intuitive visual indicators

## Technical Stack

- **Frontend**: HTML5, CSS3 with Bootstrap, Vanilla JavaScript
- **Backend**: Python with Flask
- **Machine Learning**: TensorFlow/Keras LSTM neural network
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Dataset**: Kaggle's fake-and-real-news-dataset

## Model Information

The fake news detection model is an LSTM (Long Short-Term Memory) neural network with the following characteristics:

- **Architecture**: Deep bidirectional LSTM
- **Training Dataset**: Kaggle fake-and-real-news-dataset
- **Text Representation**: Word embeddings
- **Performance**: ~93% accuracy on validation data

## Setup and Installation

1. Clone the repository
2. Install dependencies: 
   ```
   pip install tensorflow flask pandas numpy scikit-learn kagglehub
   ```
3. Run the application:
   ```
   python main.py
   ```
4. Open your browser and navigate to `http://localhost:5000`

## Usage

### Single Text Analysis

1. Enter or paste news text into the text area
2. Click "Analyze"
3. View the results showing prediction and confidence score

### Batch Analysis

1. Prepare a CSV file with a 'text' column containing news articles
2. Click "Choose File" in the Batch Analysis section
3. Select your CSV file
4. Click "Upload & Analyze"
5. View the summary and individual results

## Limitations

- The model is trained on specific types of news articles and may not perform well on very specialized content
- Short texts may not provide enough context for accurate prediction
- The model cannot fact-check specific claims, but rather identifies patterns associated with fake news
- Always verify information from multiple reliable sources

## Future Improvements

- Integration with fact-checking APIs
- URL input for direct analysis of news websites
- Multi-language support
- User accounts for personalized history
- Advanced metrics and explanations for predictions

## Disclaimer

This tool is provided for informational purposes only. While it uses machine learning to identify potential fake news, it is not infallible. Always verify information from multiple reliable sources.
