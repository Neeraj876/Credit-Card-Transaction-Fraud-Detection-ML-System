# data_ingest.py
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import SerializationContext, MessageField
from bson import ObjectId
import datetime
from time import sleep

# Configuration
MONGO_URI = "mongodb+srv://neerajjj6785:Admin123@cluster0.maegd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "FRAUD"
COLLECTION_NAME = "creditcardData"
KAFKA_BROKER = "3.80.145.193:9092"
RAW_TOPIC = "raw_transactions"
SCHEMA_REGISTRY_URL = "http://3.80.145.193:8081"
SCHEMA_SUBJECT = "raw_transactions-value"


def convert_document(doc):
    """Convert MongoDB document to match Avro schema"""
    
    def parse_date(value, default):
        """Ensure date values are in ISO format"""
        if isinstance(value, datetime.datetime):  
            return value.isoformat()  
        elif isinstance(value, str):  # If already a string, return as is
            return value  
        return default.isoformat()

    return {
        "_id": str(doc.get("_id", ObjectId())),
        "Unnamed_0": int(doc.get("Unnamed_0", 0)),
        "trans_date_trans_time": parse_date(doc.get("trans_date_trans_time"), datetime.datetime.now()),
        "cc_num": int(doc.get("cc_num", 0)),
        "merchant": str(doc.get("merchant", "")),
        "category": str(doc.get("category", "")),
        "amt": float(doc.get("amt", 0.0)),
        "first": str(doc.get("first", "")),
        "last": str(doc.get("last", "")),
        "gender": str(doc.get("gender", "")),
        "street": str(doc.get("street", "")),
        "city": str(doc.get("city", "")),
        "state": str(doc.get("state", "")),
        "zip": int(doc.get("zip", 0)),
        "lat": float(doc.get("lat", 0.0)),
        "long": float(doc.get("long", 0.0)),
        "city_pop": int(doc.get("city_pop", 0)),
        "job": str(doc.get("job", "")),
        "dob": parse_date(doc.get("dob"), datetime.datetime(1970, 1, 1)),
        "trans_num": str(doc.get("trans_num", "")),
        "unix_time": int(doc.get("unix_time", 0)),
        "merch_lat": float(doc.get("merch_lat", 0.0)),
        "merch_long": float(doc.get("merch_long", 0.0)),
        "is_fraud": int(doc.get("is_fraud", 0))
    }

# MongoDB Connection
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    print("✅ MongoDB connected.")
except ServerSelectionTimeoutError:
    print("❌ MongoDB connection failed.")
    exit(1)

# Schema Registry Setup
schema_registry_client = SchemaRegistryClient({"url": SCHEMA_REGISTRY_URL})
try:
    schema_response = schema_registry_client.get_latest_version(SCHEMA_SUBJECT)
    avro_serializer = AvroSerializer(schema_registry_client, schema_response.schema.schema_str)
except Exception as e:
    print(f"❌ Schema Registry error: {e}")
    exit(1)

# Kafka Producer
producer = Producer({
    "bootstrap.servers": KAFKA_BROKER,
    "message.max.bytes": 5242880,
    "acks": "all"
})

def produce_to_kafka():
    while True:
        try:
            with collection.watch(full_document='updateLookup') as stream:
                print("🔄 Listening for MongoDB changes...")
                for change in stream:
                    if change["operationType"] in ["insert", "replace", "update"]:
                        raw_doc = change.get("fullDocument", {})
                        converted = convert_document(raw_doc)

                        # Debugging: Print the raw document
                        print(f"🔍 MongoDB Document: {converted}")
                        
                        try:
                            avro_data = avro_serializer(
                                converted,
                                SerializationContext(RAW_TOPIC, MessageField.VALUE)
                            )
                            # Produce to raw topic
                            producer.produce(RAW_TOPIC, avro_data)
                            producer.poll(0)
                            print(f"✅ Produced to {RAW_TOPIC}: {converted['trans_num']}")
                        except Exception as e:
                            print(f"❌ Serialization error: {e}")

        except Exception as e:
            print(f"⚠️ MongoDB connection error: {e}")
            sleep(5)

if __name__ == "__main__":
    produce_to_kafka()
