import re

def clean_text(text):
    """
    Clean and normalize the text
    
    Args:
        text: Raw text string
    
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase (as mentioned in the model description)
    text = text.lower()
    
    # Remove newline characters (as mentioned in the model description)
    text = re.sub(r'\n', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    """
    Preprocess text for the model
    
    Args:
        text: Raw text string
    
    Returns:
        str: Cleaned text ready for vectorization
    """
    # Clean the text
    cleaned_text = clean_text(text)
    
    return cleaned_text
