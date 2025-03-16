from feast import FeatureView, Entity, ValueType, PushSource, Field
from feast.types import Float64, Int64, String, PrimitiveFeastType
from datetime import timedelta
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource

# Entity: Unique Transaction Identifier
transaction_entity = Entity(
    name="transaction_id",
    join_keys=["transaction_id"],
    value_type=ValueType.INT64,
    description="Unique transaction identifier",
)

# PostgreSQL Source
transaction_batch_source = PostgreSQLSource(
    name="transaction_batch_source",
    query="SELECT transaction_id, event_timestamp, cc_num, amt, merchant, category, gender, job, trans_year, trans_month, is_fraud, city_pop FROM transactions_raw",
    timestamp_field="event_timestamp",
    created_timestamp_column=None, 
 )

# Push Source (for real-time updates)
transaction_push_source = PushSource(
    name="transaction_push_source",
    batch_source=transaction_batch_source,  # Linking to batch source
)

# Feature View
creditcard_fraud_fv = FeatureView(
    name="creditcard_fraud",
    entities=[transaction_entity],
    online=True,
    ttl=timedelta(weeks=520),
    schema=[
        Field(name="cc_num", dtype=Int64),
        Field(name="amt", dtype=Float64),  
        Field(name="merchant", dtype=String),
        Field(name="category", dtype=String),
        Field(name="gender", dtype=String),
        Field(name="job", dtype=String),
        Field(name="trans_year", dtype=Int64),
        Field(name="trans_month", dtype=Int64),
        Field(name="is_fraud", dtype=Int64),
        Field(name="city_pop", dtype=Int64),
    ],
    source=transaction_batch_source,
)