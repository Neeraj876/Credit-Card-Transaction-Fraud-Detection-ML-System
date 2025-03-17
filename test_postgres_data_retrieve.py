import feast
from feast import FeatureStore
import pandas as pd

# Initialize Feature Store
store = FeatureStore(repo_path="/mnt/d/real_time_streaming/my_feature_repo/feature_repo")

# Define the list of features you want to fetch
features = [
    "creditcard_fraud:cc_num", 
    "creditcard_fraud:amt", 
    "creditcard_fraud:merchant", 
    "creditcard_fraud:category", 
    "creditcard_fraud:gender", 
    "creditcard_fraud:job", 
    "creditcard_fraud:trans_year", 
    "creditcard_fraud:trans_month", 
    "creditcard_fraud:is_fraud", 
    "creditcard_fraud:city_pop"
]

# Entity DataFrame (PostgreSQL Query)
entity_df = store.get_historical_features(
    features=features, 
    entity_df="SELECT transaction_id, event_timestamp FROM transactions_raw"
).to_df()

# Save the DataFrame to CSV
entity_df.to_csv("training_features.csv", index=False)

# Display results
print("Saved the requested features to 'output_features.csv'")

