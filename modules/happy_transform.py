
import tensorflow as tf
LABEL_KEY = "happy"
FEATURE_KEYS = [
    "infoavail",
    "housecost",
    "schoolquality",
    "policetrust",
    "streetquality",
    "events"
]
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"
def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    
    """
    
    outputs = {}
    
   # Loop through the feature columns and apply transformations (e.g., scaling or casting)
    for feature_key in FEATURE_KEYS:
        outputs[transformed_name(feature_key)] = tf.cast(inputs[feature_key], tf.float32)
    
    # Transforming the 'FetalHealth' label into integer type if it's not already
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
