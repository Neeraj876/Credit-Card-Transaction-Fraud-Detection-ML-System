from confluent_kafka import Consumer, Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
from confluent_kafka.serialization import SerializationContext, MessageField
from observability.alerts.alert import send_alert_to_elasticsearch, send_alert_to_alertmanager
from src.logging.otel_logger import logger
import json
# import logging
import sys
import time

# Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("transaction_validator.log"),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger("transaction_validator")

# Configuration
CONFIG = {
    "kafka_broker": "3.87.22.62:9092",
    "schema_registry_url": "http://3.87.22.62:8081",
    "topics": {
        "input": "raw_transactions",
        "valid": "valid_transactions",
        "invalid": "invalid_transactions"
    },
    "group_id": "fraud_validation_group"
}

logger.info("Starting transaction validation service")
logger.debug(f"Configuration: {CONFIG}")

# Initialize Kafka consumer
try:
    consumer_config = {
        "bootstrap.servers": CONFIG["kafka_broker"],
        "group.id": CONFIG["group_id"],
        "auto.offset.reset": "earliest"
    }
    consumer = Consumer(consumer_config)
    consumer.subscribe([CONFIG["topics"]["input"]])
    logger.info(f"Consumer initialized and subscribed to {CONFIG['topics']['input']}")
except Exception as e:
    logger.error(f"Failed to initialize consumer: {str(e)}")
    sys.exit(1)

# Initialize Kafka producer
try:
    producer = Producer({"bootstrap.servers": CONFIG["kafka_broker"]})
    logger.info("Producer initialized")
except Exception as e:
    logger.error(f"Failed to initialize producer: {str(e)}")
    sys.exit(1)

# Initialize Schema Registry
try:
    schema_registry_client = SchemaRegistryClient({"url": CONFIG["schema_registry_url"]})
    schema = schema_registry_client.get_latest_version("raw_transactions-value").schema.schema_str
    avro_deserializer = AvroDeserializer(schema_registry_client, schema)
    logger.info("Schema Registry client initialized")
    logger.debug(f"Using schema: {schema}")
except Exception as e:
    logger.error(f"Failed to initialize Schema Registry client: {str(e)}")
    sys.exit(1)

# Delivery callback for producer
def delivery_report(err, msg):
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

def validate_data(record):
    """Apply validation rules for transactions based on the actual schema fields."""
    logger.debug(f"Validating record: {record}")
    
    # Using actual fields from the schema
    # trans_num is the transaction ID
    # amt is the amount
    has_transaction_id = "trans_num" in record and record["trans_num"]
    valid_amount = "amt" in record and record["amt"] > 0
    
    valid = has_transaction_id and valid_amount
    
    if not has_transaction_id:
        logger.warning("Validation failed: Missing trans_num (transaction ID)")
    if not valid_amount:
        amt_value = record.get("amt", "not present")
        logger.warning(f"Validation failed: Invalid amt (amount) {amt_value}")
    
    # Log additional details about the transaction for debugging
    transaction_detail = {
        "transaction_id": record.get("trans_num", "N/A"),
        "amount": record.get("amt", "N/A"),
        "date_time": record.get("trans_date_trans_time", "N/A"),
        "merchant": record.get("merchant", "N/A"),
        "category": record.get("category", "N/A"),
        "is_fraud": record.get("is_fraud", "N/A")
    }
    logger.debug(f"Transaction details: {transaction_detail}")
    
    return valid

# Process messages
message_count = 0
valid_count = 0
invalid_count = 0
start_time = time.time()
last_log_time = start_time

logger.info("Starting message processing loop")
try:
    while True:
        msg = consumer.poll(1.0)
        
        # Print stats every 100 messages
        # if message_count > 0 and message_count % 100 == 0:
        #     elapsed = time.time() - start_time
        #     logger.info(f"Stats: Processed {message_count} messages ({valid_count} valid, {invalid_count} invalid) in {elapsed:.2f} seconds")
        
        if msg is None:
            continue
            
        if msg.error():
            logger.error(f"Consumer error: {msg.error()}")
            continue
        
        try:
            message_count += 1
            key = msg.key().decode("utf-8") if msg.key() else None
            topic = CONFIG["topics"]["input"]
            logger.debug(f"Processing message with key: {key} from topic: {topic}")
            
            # Create serialization context with topic information
            ctx = SerializationContext(topic, MessageField.VALUE)
            
            # Deserialize with context
            record = avro_deserializer(msg.value(), ctx)
            logger.debug(f"Deserialized record: {record}")
            
            # Validate the record using actual schema fields
            is_valid = validate_data(record)
            
            if is_valid:
                valid_count += 1
                producer.produce(
                    CONFIG["topics"]["valid"],
                    key=key,
                    value=json.dumps(record),
                    callback=delivery_report
                )
                logger.info(f"Valid transaction {record.get('trans_num')} sent to valid_transactions")
            else:
                invalid_count += 1
                producer.produce(
                    CONFIG["topics"]["invalid"],
                    key=key,
                    value=json.dumps(record),
                    callback=delivery_report
                )
                logger.info(f"Invalid transaction {record.get('trans_num', 'unknown')} sent to invalid_transactions")

                # Send alert to Elasticsearch when validation fails
                error_message = f"Invalid transaction {record.get('trans_num', 'unknown')} - Invalid amount or missing transaction ID."
                send_alert_to_elasticsearch(error_message)
                send_alert_to_alertmanager(error_message)
                        
            producer.poll(0)  # Trigger delivery reports

            # Calculate records per second (RPS)
            elapsed_time = time.time() - last_log_time
            if elapsed_time >= 1.0:  # Log every second
                rps = message_count / (time.time() - start_time)
                logger.info(f"Processed {message_count} messages | RPS: {rps:.2f} | Valid: {valid_count} | Invalid: {invalid_count}")
                last_log_time = time.time()
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
    
        producer.flush()

except KeyboardInterrupt:
    logger.info("Shutting down gracefully...")
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}", exc_info=True)
finally:
    # Print final stats
    elapsed = time.time() - start_time
    rps = message_count / elapsed if elapsed > 0 else 0
    logger.info(f"Final stats: Processed {message_count} messages | RPS: {rps:.2f} | Valid: {valid_count} | Invalid: {invalid_count} | Time: {elapsed:.2f}s")
    
    # Clean up resources
    consumer.close()
    logger.info("Consumer closed")
    logger.info("Service stopped")