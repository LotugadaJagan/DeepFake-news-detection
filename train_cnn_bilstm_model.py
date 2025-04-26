import os
import re
import pandas as pd
import numpy as np
import json
import pickle
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, SpatialDropout1D, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available. Using CNN-BiLSTM model.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow is not available. Falling back to scikit-learn models.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

# Define paths
MODELS_DIR = 'models'
TF_MODEL_PATH = os.path.join(MODELS_DIR, 'fake_news_cnn_bilstm.h5')
TOKENIZER_PATH = os.path.join(MODELS_DIR, 'tokenizer.pickle')
MODEL_INFO_PATH = os.path.join(MODELS_DIR, 'model_info.json')
HISTORY_PATH = os.path.join(MODELS_DIR, 'training_history.pickle')
FALLBACK_MODEL_PATH = os.path.join(MODELS_DIR, 'fake_news_model.pkl')
FALLBACK_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer.pkl')

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def clean_text(text):
    """Clean and normalize the text"""
    text = str(text).lower()  # Lowercase
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text

def load_and_prepare_data(fake_path, true_path):
    """
    Load and prepare the Kaggle fake news dataset
    
    Args:
        fake_path: Path to the Fake.csv file
        true_path: Path to the True.csv file
    
    Returns:
        Various processed datasets and data information
    """
    logger.info(f"Loading data from {fake_path} and {true_path}")
    
    # Load the datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Add labels (fake = 0, true = 1)
    fake_df['label'] = 0
    true_df['label'] = 1
    
    # Combine the datasets
    data = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Preprocess the text
    logger.info("Preprocessing text...")
    data['cleaned_text'] = data['text'].apply(clean_text)
    
    # Split the data
    X = data['cleaned_text']
    y = data['label']
    
    # Train-validation-test split (80-10-10)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    logger.info(f"Data split: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, data

def create_tensorflow_dataset(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Create TensorFlow-ready datasets with tokenization and padding
    """
    # Text vectorization parameters
    max_words = 5000  # Keep most frequent 5000 words
    max_len = 300     # Maximum article length
    
    # Create tokenizer
    logger.info("Creating tokenizer...")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    train_sequences = tokenizer.texts_to_sequences(X_train)
    val_sequences = tokenizer.texts_to_sequences(X_val)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences to uniform length
    X_train_pad = pad_sequences(train_sequences, maxlen=max_len, padding='post')
    X_val_pad = pad_sequences(val_sequences, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(test_sequences, maxlen=max_len, padding='post')
    
    # Save tokenizer for future use
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    return X_train_pad, X_val_pad, X_test_pad, tokenizer, max_words, max_len

def build_cnn_bilstm_model(max_words, max_len, embedding_dim=100):
    """
    Build a hybrid CNN-BiLSTM model for text classification
    """
    logger.info("Building CNN-BiLSTM model...")
    model = Sequential([
        # Embedding layer (will be initialized randomly as we don't have GloVe)
        Embedding(input_dim=max_words,
                output_dim=embedding_dim,
                input_length=max_len),
        
        SpatialDropout1D(0.3),  # Special dropout for embeddings
        
        # CNN for local feature extraction
        Conv1D(64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        
        # BiLSTM for context understanding
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        
        # Classification layers
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    model.summary()
    return model

def train_tensorflow_model(model, X_train_pad, y_train, X_val_pad, y_val, epochs=3):
    """
    Train the CNN-BiLSTM model
    """
    logger.info(f"Training CNN-BiLSTM model for {epochs} epochs...")
    history = model.fit(
        X_train_pad, y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(X_val_pad, y_val),
        verbose=1
    )
    
    # Save training history
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    
    return model, history

def evaluate_tensorflow_model(model, X_test_pad, y_test):
    """
    Evaluate the CNN-BiLSTM model
    """
    logger.info("Evaluating CNN-BiLSTM model...")
    y_pred_prob = model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return accuracy, precision, recall, f1, cm, y_pred

def save_fallback_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Create and save a scikit-learn model as fallback
    """
    logger.info("Creating fallback scikit-learn model...")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train a logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Fallback model metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Save the model and vectorizer
    with open(FALLBACK_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    with open(FALLBACK_VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return accuracy, precision, recall, f1, y_pred, vectorizer, model

def save_model_info(accuracy, precision, recall, f1, training_time, data, model_type="CNN-BiLSTM"):
    """
    Save model information to JSON file
    """
    # Get counts from the original full dataset
    total_fake = len(data[data['label'] == 0])
    total_true = len(data[data['label'] == 1])
    total_samples = len(data)
    
    # Create model info
    model_info = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'num_samples': total_samples,
        'fake_samples': int(total_fake),
        'real_samples': int(total_true),
        'architecture': model_type,
        'vectorizer': 'Word Embeddings + CNN + BiLSTM' if model_type == "CNN-BiLSTM" else 'TF-IDF',
        'embedding': 'Learned Embeddings (100 dimensions)',
        'preprocessing': 'Lowercase, removing special characters, train-val-test split (80-10-10)',
        'model_structure': 'Embedding Layer → Conv1D + MaxPooling → Bidirectional LSTM → Dense Layers',
        'training_time': float(training_time),
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'note': 'Model trained on Kaggle fake-and-real-news-dataset. CNN-BiLSTM architecture for enhanced feature extraction.'
    }
    
    logger.info(f"Saving model info to {MODEL_INFO_PATH}")
    with open(MODEL_INFO_PATH, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    return model_info

def plot_training_history(history):
    """
    Plot the training history graphs
    """
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'))
    logger.info(f"Training history graph saved to {os.path.join(MODELS_DIR, 'training_history.png')}")

def plot_confusion_matrix(cm, y_test):
    """
    Plot the confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Fake (0)', 'Real (1)'])
    plt.yticks([0.5, 1.5], ['Fake (0)', 'Real (1)'])
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'))
    logger.info(f"Confusion matrix saved to {os.path.join(MODELS_DIR, 'confusion_matrix.png')}")

def main():
    """Main function to train and save the model"""
    # Check if the CSV files exist
    fake_path = 'dataset/Fake_small.csv'
    true_path = 'dataset/True_small.csv'
    
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        logger.error(f"Dataset files not found at {fake_path} or {true_path}")
        return
    
    # Start timer
    start_time = time.time()
    
    # Load and prepare the data
    X_train, X_val, X_test, y_train, y_val, y_test, data = load_and_prepare_data(fake_path, true_path)
    
    # Check if TensorFlow is available for CNN-BiLSTM
    if TENSORFLOW_AVAILABLE:
        # Prepare data for TensorFlow
        X_train_pad, X_val_pad, X_test_pad, tokenizer, max_words, max_len = create_tensorflow_dataset(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Build the model
        model = build_cnn_bilstm_model(max_words, max_len)
        
        # Train the model
        model, history = train_tensorflow_model(model, X_train_pad, y_train, X_val_pad, y_val, epochs=3)
        
        # Evaluate the model
        accuracy, precision, recall, f1, cm, y_pred = evaluate_tensorflow_model(model, X_test_pad, y_test)
        
        # Save the model
        model.save(TF_MODEL_PATH)
        logger.info(f"CNN-BiLSTM model saved to {TF_MODEL_PATH}")
        
        # Plot training history
        plot_training_history(history)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, y_test)
        
        # Also create a fallback scikit-learn model
        save_fallback_model(X_train, X_val, X_test, y_train, y_val, y_test)
        
    else:
        # If TensorFlow is not available, use scikit-learn model
        accuracy, precision, recall, f1, y_pred, vectorizer, model = save_fallback_model(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f"Total training time: {training_time:.2f} seconds")
    
    # Save model info
    model_type = "CNN-BiLSTM" if TENSORFLOW_AVAILABLE else "Logistic Regression"
    save_model_info(accuracy, precision, recall, f1, training_time, data, model_type)
    
    logger.info("Model training and saving completed successfully")

if __name__ == "__main__":
    main()