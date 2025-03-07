from feast import Entity, FeatureView, Field, PushSource, ValueType
from feast.types import Float32, Int64, String
from datetime import timedelta
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)

# --- DEFINE ENTITY ---
transaction_entity = Entity(
    name="transaction_id",
    value_type=ValueType.INT64,
    description="A unique transaction identifier",
)

# --- OFFLINE STORE (POSTGRESQL) ---
transaction_batch_source = PostgreSQLSource(
    name="transaction_batch_source",
    query="SELECT transaction_id, cc_num, amt, is_fraud, trans_month, trans_year, "
          "merchant, category, gender, city_pop, job, event_timestamp, created_timestamp "
          "FROM public.transactions_raw",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# --- PUSH SOURCE (FOR REAL-TIME UPDATES TO REDIS) ---
transaction_push_source = PushSource(
    name="transaction_push_source",
    batch_source=transaction_batch_source, # Link to the batch source
)

# --- DEFINE FEATURE VIEW ---
fraud_feature_view = FeatureView(
    name="fraud_features",
    entities=[transaction_entity],
    ttl=timedelta(days=1),
    online=True,
    source=transaction_push_source,
    schema=[
        Field(name="cc_num", dtype=Int64),
        Field(name="amt", dtype=Float32),
        Field(name="is_fraud", dtype=Int64),
        Field(name="trans_month", dtype=Int64),
        Field(name="trans_year", dtype=Int64),
        Field(name="merchant", dtype=String),
        Field(name="category", dtype=String),
        Field(name="gender", dtype=String),
        Field(name="city_pop", dtype=Int64),
        Field(name="job", dtype=String),
    ],
    tags={
        "domain": "fraud_detection",
    },
)