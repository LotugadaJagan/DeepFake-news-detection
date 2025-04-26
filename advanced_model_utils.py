import os
import re
import pickle
import logging
import numpy as np
from text_preprocessing import preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
MODELS_DIR = 'models'
TF_MODEL_PATH = os.path.join(MODELS_DIR, 'fake_news_cnn_bilstm.h5')
TOKENIZER_PATH = os.path.join(MODELS_DIR, 'tokenizer.pickle')
FALLBACK_MODEL_PATH = os.path.join(MODELS_DIR, 'fake_news_model.pkl')
FALLBACK_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer.pkl')

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available. Using CNN-BiLSTM model.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow is not available. Falling back to scikit-learn models.")

def load_models():
    """
    Load the appropriate models based on availability
    """
    if TENSORFLOW_AVAILABLE and os.path.exists(TF_MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        # Load TensorFlow CNN-BiLSTM model
        logger.info(f"Loading CNN-BiLSTM model from {TF_MODEL_PATH}")
        tf_model = load_model(TF_MODEL_PATH)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {TOKENIZER_PATH}")
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        
        return tf_model, tokenizer, None, None, True
    
    elif os.path.exists(FALLBACK_MODEL_PATH) and os.path.exists(FALLBACK_VECTORIZER_PATH):
        # Load fallback scikit-learn model
        logger.info(f"Loading fallback model from {FALLBACK_MODEL_PATH}")
        with open(FALLBACK_MODEL_PATH, 'rb') as f:
            fallback_model = pickle.load(f)
        
        logger.info(f"Loading vectorizer from {FALLBACK_VECTORIZER_PATH}")
        with open(FALLBACK_VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return None, None, fallback_model, vectorizer, False
    
    else:
        logger.error("No models found. Please train the models first.")
        return None, None, None, None, False

def predict_with_cnn_bilstm(model, tokenizer, text, max_len=300):
    """
    Make a prediction using the CNN-BiLSTM model
    
    Args:
        model: TensorFlow CNN-BiLSTM model
        tokenizer: Keras tokenizer
        text: Raw text input
        max_len: Maximum sequence length
    
    Returns:
        is_fake (bool): True if fake news, False if real news
        confidence (float): Confidence score
    """
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    
    # Make prediction
    prediction = model.predict(padded)[0][0]
    
    # In the CNN-BiLSTM model, 1 is real and 0 is fake
    # So if prediction > 0.5, it's likely real (not fake)
    is_fake = prediction < 0.5
    
    # Calculate confidence (how certain the model is)
    confidence = 1 - prediction if is_fake else prediction
    
    return is_fake, float(confidence)

def predict_with_fallback(model, vectorizer, text):
    """
    Make a prediction using the fallback scikit-learn model
    
    Args:
        model: Scikit-learn model
        vectorizer: TF-IDF vectorizer
        text: Raw text input
    
    Returns:
        is_fake (bool): True if fake news, False if real news
        confidence (float): Confidence score
    """
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Vectorize the text
    text_vector = vectorizer.transform([preprocessed_text])
    
    # Make prediction
    probas = model.predict_proba(text_vector)[0]
    
    # Determine if fake or real
    # Class 1 is real, class 0 is fake
    is_fake = model.predict(text_vector)[0] == 0
    
    # Calculate confidence based on probability of the predicted class
    confidence = float(probas[0] if is_fake else probas[1])
    
    return is_fake, confidence

def predict_news(text):
    """
    Make prediction on news text using the best available model
    
    Args:
        text: Raw text string
    
    Returns:
        tuple: (prediction boolean, confidence score)
    """
    # Load models
    tf_model, tokenizer, fallback_model, vectorizer, use_tf = load_models()
    
    if use_tf:
        # Use CNN-BiLSTM model
        return predict_with_cnn_bilstm(tf_model, tokenizer, text)
    else:
        # Use fallback scikit-learn model
        return predict_with_fallback(fallback_model, vectorizer, text)

if __name__ == "__main__":
    # Test with an example
    sample_text = "Scientists discover new species in deep ocean trench"
    is_fake, confidence = predict_news(sample_text)
    result = "FAKE" if is_fake else "REAL"
    logger.info(f"Prediction: {result} with {confidence:.2%} confidence")