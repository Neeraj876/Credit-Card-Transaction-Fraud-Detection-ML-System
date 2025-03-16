from feast import Entity, FeatureStore, ValueType

# Define the entity
transaction_id = Entity(name="transaction_id", value_type=ValueType.INT64)

# Define the FeatureStore (Ensure this is the correct path to your feature repo)
store = FeatureStore(repo_path="/mnt/d/real_time_streaming/my_feature_repo/feature_repo")

# Fetch features using transaction_id as the entity
feature_vector = store.get_online_features(
    features=[
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
    ],
    entity_rows=[{"transaction_id": 92001998}] 
).to_df()

print(feature_vector)