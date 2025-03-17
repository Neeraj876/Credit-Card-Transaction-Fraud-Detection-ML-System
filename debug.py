import numpy as np
import pandas as pd
import logging
from joblib import load

# Load preprocessor and model (update paths if needed)
PREPROCESSOR_PATH = "final_model/preprocessor.pkl"
MODEL_PATH = "final_model/model.pkl"

try:
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading preprocessor and model...")
    preprocessor = load(PREPROCESSOR_PATH)
    model = load(MODEL_PATH)

    logging.info("Preprocessor and model loaded successfully.")

    # Define raw test data
    raw_data = {
    "transaction_id": [109001662],
    "cc_num": [3560725013359375],
    "amt": [24.84000015258789],
    "merchant": ["fraud_Hamill-D'Amore"],
    "category": ["health_fitness"],
    "gender": ["F"],
    "job": ["Cytogeneticist"],
    "trans_year": [2020],
    "trans_month": [6],
    "city_pop": [23]  
}


    # Convert to DataFrame
    feature_df = pd.DataFrame(raw_data)
    logging.info(f"Raw feature DataFrame:\n{feature_df}")

    # Drop 'transaction_id' as it's not needed for model input
    feature_df.drop(columns=['transaction_id'], inplace=True, errors='ignore')

    # Ensure categorical features are strings
    cat_columns = ["merchant", "category", "gender", "job"]
    for col in cat_columns:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].astype(str)

    # Print data types before transformation
    logging.info(f"Feature DataFrame after type conversion:\n{feature_df.dtypes}")

    # Apply preprocessor transformation
    preprocessed_features = preprocessor.transform(feature_df)

    # Convert sparse matrix to dense array if needed
    if hasattr(preprocessed_features, "toarray"):
        preprocessed_features = preprocessed_features.toarray()

    logging.info(f"Preprocessed feature shape: {preprocessed_features.shape}")

    # Ensure model is loaded
    if model is None:
        raise Exception("Model is not loaded correctly.")

    # Get model prediction
    if hasattr(model, "predict_proba"):
        fraud_probability = model.predict_proba(preprocessed_features)[:, 1][0]
    else:
        fraud_probability = model.predict(preprocessed_features)[0]

    # Determine fraud label
    fraud_label = int(fraud_probability >= 0.5)

    logging.info(f"Fraud Probability: {fraud_probability}")
    logging.info(f"Fraud Label: {fraud_label}")

    print(f"Final Prediction Output: \n"
          f"Fraud Probability: {fraud_probability:.4f}\n"
          f"Fraud Label: {fraud_label}")

except Exception as e:
    logging.error(f"Error during debugging: {str(e)}")
