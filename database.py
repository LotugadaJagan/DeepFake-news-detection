import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import time
from datetime import datetime

logger = logging.getLogger(__name__)

def get_db_connection():
    """Get a database connection"""
    conn = psycopg2.connect(os.environ.get("DATABASE_URL"))
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            prediction BOOLEAN NOT NULL,
            confidence REAL NOT NULL,
            timestamp TIMESTAMP NOT NULL
        )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
    finally:
        conn.close()

# Initialize database on module import
init_db()

def store_prediction(text, prediction, confidence):
    """
    Store a prediction in the database
    
    Args:
        text: News text
        prediction: Boolean indicating if the news is fake (True) or real (False)
        confidence: Confidence score of the prediction
    
    Returns:
        int: ID of the inserted prediction
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Store only the first 1000 characters of the text to save space
        truncated_text = text[:1000] + '...' if len(text) > 1000 else text
        
        # Insert the prediction
        timestamp = datetime.now()
        cursor.execute(
            'INSERT INTO predictions (text, prediction, confidence, timestamp) VALUES (%s, %s, %s, %s) RETURNING id',
            (truncated_text, prediction, confidence, timestamp)
        )
        
        # Get the ID of the inserted row
        result = cursor.fetchone()
        prediction_id = result[0] if result else None
        
        conn.commit()
        
        return prediction_id
    
    except Exception as e:
        logger.error(f"Error storing prediction: {str(e)}")
        return None
    
    finally:
        conn.close()

def get_recent_predictions(count=5):
    """
    Get the most recent predictions
    
    Args:
        count: Number of predictions to retrieve
    
    Returns:
        list: List of prediction dictionaries
    """
    conn = get_db_connection()
    try:
        # Create cursor that returns dictionaries
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get the most recent predictions
        cursor.execute(
            'SELECT id, text, prediction, confidence, timestamp FROM predictions ORDER BY id DESC LIMIT %s',
            (count,)
        )
        
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        predictions = []
        for row in rows:
            predictions.append({
                'id': row['id'],
                'text': row['text'],
                'prediction': 'fake' if row['prediction'] else 'real',
                'confidence': row['confidence'],
                'timestamp': row['timestamp'].isoformat() if row['timestamp'] else None
            })
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error retrieving recent predictions: {str(e)}")
        return []
    
    finally:
        conn.close()
