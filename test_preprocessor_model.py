#!/usr/bin/env python
"""
Independent test script for verifying model and preprocessor loading from MLflow.
This script can be run separately from the main application to verify that
the models are correctly registered and accessible.
"""

import os
import sys
import time
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import pandas as pd
import numpy as np

def main():
    """Test MLflow model and preprocessor loading independently."""
    # Start timing
    start_time = time.time()
    
    print("\n===== MLflow Model Loading Test Script =====\n")
    
    # Load environment variables
    load_dotenv()
    print("✓ Environment variables loaded")
    
    # Get MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        print("❌ Error: MLFLOW_TRACKING_URI environment variable not set")
        sys.exit(1)
    
    print(f"✓ Found MLflow tracking URI: {mlflow_uri}")
    
    # Set MLflow tracking URI
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        print("✓ MLflow tracking URI set successfully")
    except Exception as e:
        print(f"❌ Failed to set MLflow tracking URI: {str(e)}")
        sys.exit(1)
    
    # Create MLflow client
    try:
        client = MlflowClient()
        print("✓ MLflow client created successfully")
    except Exception as e:
        print(f"❌ Failed to create MLflow client: {str(e)}")
        sys.exit(1)
    
    # Get information about registered models
    try:
        model_names = [m.name for m in client.search_registered_models()]
        print(f"✓ Found registered models: {model_names}")
        
        if 'XGBClassifier' not in model_names or 'feature_preprocessor' not in model_names:
            print("❌ Error: Required models not found in the registry")
            missing = []
            if 'XGBClassifier' not in model_names:
                missing.append('XGBClassifier')
            if 'feature_preprocessor' not in model_names:
                missing.append('feature_preprocessor')
            print(f"   Missing: {', '.join(missing)}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to retrieve registered models: {str(e)}")
        sys.exit(1)
    
    # Get model versions
    model_version = None
    preprocessor_version = None
    
    try:
        # Get XGBClassifier versions
        model_versions = client.search_model_versions("name='XGBClassifier'")
        if not model_versions:
            print("❌ Error: No versions found for XGBClassifier")
            sys.exit(1)
        
        # Sort by version number (latest first)
        model_versions = sorted(model_versions, key=lambda x: int(x.version), reverse=True)
        model_version = model_versions[0].version
        print(f"✓ Latest XGBClassifier version: {model_version}")
        
        # Get preprocessor versions
        preprocessor_versions = client.search_model_versions("name='feature_preprocessor'")
        if not preprocessor_versions:
            print("❌ Error: No versions found for feature_preprocessor")
            sys.exit(1)
        
        # Sort by version number (latest first)
        preprocessor_versions = sorted(preprocessor_versions, key=lambda x: int(x.version), reverse=True)
        preprocessor_version = preprocessor_versions[0].version
        print(f"✓ Latest feature_preprocessor version: {preprocessor_version}")
    except Exception as e:
        print(f"❌ Failed to retrieve model versions: {str(e)}")
        sys.exit(1)
    
    # Load preprocessor
    try:
        print("\nLoading feature preprocessor...")
        preprocessor = mlflow.pyfunc.load_model(f"models:/feature_preprocessor/{preprocessor_version}")
        print(f"✓ Successfully loaded preprocessor (version {preprocessor_version})")
        print(f"   Type: {type(preprocessor).__name__}")
    except Exception as e:
        print(f"❌ Failed to load preprocessor: {str(e)}")
        sys.exit(1)
    
    # Load model
    try:
        print("\nLoading XGBClassifier model...")
        model = mlflow.pyfunc.load_model(f"models:/XGBClassifier/{model_version}")
        print(f"✓ Successfully loaded model (version {model_version})")
        print(f"   Type: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        sys.exit(1)
    
    # Create sample data to test model pipeline
    print("\nTesting model and preprocessor functionality with sample data...")
    try:
        # Create synthetic test data with expected features
        # Modify this based on your model's expected input features
        sample_data = pd.DataFrame({
            "cc_num": [np.int64(1234567890123456)],  # Wrap in list
            "amt": [np.float64(100.50)],  
            "merchant": ["Amazon"],  
            "category": ["Retail"],  
            "gender": ["Male"],  
            "job": ["Software Engineer"],  
            "trans_year": [np.int64(2024)],  
            "trans_month": [np.int64(3)],  
            "city_pop": [np.int64(500000)],  
        })
                
        print("✓ Created sample test data")
        
        # Process with preprocessor
        processed_data = preprocessor.predict(sample_data)
        print("✓ Successfully transformed data with preprocessor")
        
        # Make prediction
        prediction = model.predict(processed_data)
        print("✓ Successfully made prediction with model")
        print(f"   Prediction: {prediction[0]} (0=Normal, 1=Fraud)")
        
    except Exception as e:
        print(f"❌ Error when testing model pipeline: {str(e)}")
        print("   This may indicate an issue with the preprocessor or model expectations.")
        sys.exit(1)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    print("\n===== Test Summary =====")
    print(f"✓ Overall test completed successfully in {elapsed_time:.2f} seconds")
    print(f"✓ MLflow tracking URI: {mlflow_uri}")
    print(f"✓ XGBClassifier model (version {model_version}) loaded successfully")
    print(f"✓ Preprocessor (version {preprocessor_version}) loaded successfully")
    print("✓ Model pipeline works with sample data")
    print("\nYour model and preprocessor are ready for production use!")
    
if __name__ == "__main__":
    main()