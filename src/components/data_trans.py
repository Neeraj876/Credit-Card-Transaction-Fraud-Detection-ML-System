import os 
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, month, year, expr, unix_timestamp, when, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType, FloatType, TimestampType

from src.logging.logger import logging

# Configuration
CONFIG = {
    "kafka": {
        "broker": "44.204.24.108:9092",
        "topic": "valid_transactions",
    },
    "storage": {
        "postgres_url": "jdbc:postgresql://localhost:5432/feast_db",
        "postgres_table": "transactions_raw",
        "postgres_options": {
            "user": "postgres",
            "password": "neeraj",
            "driver": "org.postgresql.Driver"
        }
    },
    "redis": {
        "host": "localhost",
        "port": "6379"
    },
    "checkpoint": "/mnt/d/real_time_streaming/checkpoint"
}

logging.info(f"Starting Spark Streaming application with config: {CONFIG}")

# Ensure checkpoint directory exists
os.makedirs(CONFIG["checkpoint"], exist_ok=True)
logging.info(f"Checkpoint directory ensured: {CONFIG['checkpoint']}")

# Define JSON Schema for Kafka messages - updated to match actual incoming data
schema = StructType([
    StructField("_id", StringType(), True),
    StructField("Unnamed_0", IntegerType(), True),  # Will be renamed to transaction_id
    StructField("trans_date_trans_time", StringType(), True),
    StructField("cc_num", LongType(), True),
    StructField("merchant", StringType(), True),
    StructField("category", StringType(), True),
    StructField("amt", DoubleType(), True),
    StructField("first", StringType(), True),
    StructField("last", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("street", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("zip", IntegerType(), True),
    StructField("lat", DoubleType(), True),
    StructField("long", DoubleType(), True),
    StructField("city_pop", IntegerType(), True),
    StructField("job", StringType(), True),
    StructField("dob", StringType(), True),
    StructField("trans_num", StringType(), True),
    StructField("unix_time", LongType(), True),
    StructField("merch_lat", DoubleType(), True),
    StructField("merch_long", DoubleType(), True),
    StructField("is_fraud", IntegerType(), True)
])

def debug_batch(batch_df, batch_id):
    """Debug function to examine incoming data structure"""
    try:
        count = batch_df.count()
        logging.info(f"Batch {batch_id}: Received {count} records")
        
        if count > 0:
            # Show schema and sample data
            logging.info(f"Batch {batch_id} Schema:")
            for field in batch_df.schema.fields:
                logging.info(f"  {field.name}: {field.dataType}")
            
            # Show sample records
            logging.info(f"Batch {batch_id} Sample Data:")
            sample_rows = batch_df.limit(2).collect()
            for i, row in enumerate(sample_rows):
                logging.info(f"  Row {i+1}: {row.asDict()}")
        else:
            logging.info(f"Batch {batch_id}: Empty batch")
            
    except Exception as e:
        logging.error(f"Error in debug_batch for batch {batch_id}: {str(e)}")

def write_to_stores(batch_df, epoch_id):
    if batch_df.isEmpty():
        logging.info(f"Epoch {epoch_id} - No data to process.")
        return

    try:
        count_before = batch_df.count()
        logging.info(f"Epoch {epoch_id} - Total rows before filtering: {count_before}")
        
        # Debug the incoming batch
        debug_batch(batch_df, epoch_id)

        # Create a unique transaction_id using multiple columns
        from pyspark.sql.functions import abs, expr, monotonically_increasing_id
        
        # Use monotonically_increasing_id() to generate guaranteed unique IDs within this batch
        # Combined with epoch_id to make it unique across batches
        batch_df = batch_df.withColumn(
            "transaction_id",
            (monotonically_increasing_id() + (epoch_id * 1000000)).cast("int")
        )
        
        # Now check if expected columns exist
        required_columns = ["transaction_id", "cc_num", "amt", "is_fraud", "event_timestamp"]
        available_columns = batch_df.columns
        
        logging.info(f"Available columns: {available_columns}")
        
        missing_columns = [col for col in required_columns if col not in available_columns]
        
        # If "event_timestamp" is missing but "trans_date_trans_time" exists, create it
        if "event_timestamp" in missing_columns and "trans_date_trans_time" in available_columns:
            batch_df = batch_df.withColumn("event_timestamp", 
                                          to_timestamp(col("trans_date_trans_time"), "yyyy-MM-dd HH:mm:ss"))
            missing_columns.remove("event_timestamp")
        
        if missing_columns:
            logging.error(f"Epoch {epoch_id} - Missing required columns: {missing_columns}")
            logging.error(f"Available columns: {available_columns}")
            return
        
        # batch_df = batch_df.withColumn(
        #     "transaction_id",
        #     (unix_timestamp(current_timestamp()) * 1000).cast("long")
        # )
        
        # Select only required features
        try:
            final_df = batch_df.select(
                col("transaction_id"),  
                col("cc_num").cast("long"),  
                col("amt").cast("float"),
                col("is_fraud").cast("integer"),
                col("event_timestamp"),
                month(col("event_timestamp")).alias("trans_month").cast("integer"),
                year(col("event_timestamp")).alias("trans_year").cast("integer"),
                col("merchant"),
                col("category"),
                col("gender"),
                col("city_pop").cast("integer"),
                col("job")
            )
        except Exception as e:
            logging.error(f"Error during column selection in epoch {epoch_id}: {str(e)}")
            # Show schema for debugging
            logging.error("Available columns and types:")
            for field in batch_df.schema.fields:
                logging.error(f"  {field.name}: {field.dataType}")
            raise

        count_after = final_df.count()
        logging.info(f"Epoch {epoch_id} - Writing {count_after} records to PostgreSQL and Redis.")

        # Debug the finalized dataframe
        debug_batch(final_df, f"{epoch_id}-final")

        # Write to PostgreSQL (Feast Offline Store)
        try:
            # CRITICAL FIX: Add these logging statements to verify the connection details
            logging.info(f"Attempting to write to PostgreSQL with URL: {CONFIG['storage']['postgres_url']}")
            logging.info(f"PostgreSQL table: {CONFIG['storage']['postgres_table']}")
            
            final_df.write \
                .format("jdbc") \
                .option("url", CONFIG["storage"]["postgres_url"]) \
                .option("dbtable", CONFIG["storage"]["postgres_table"]) \
                .option("user", CONFIG["storage"]["postgres_options"]["user"]) \
                .option("password", CONFIG["storage"]["postgres_options"]["password"]) \
                .option("driver", CONFIG["storage"]["postgres_options"]["driver"]) \
                .mode("append") \
                .save()
            logging.info(f"Epoch {epoch_id} - Successfully wrote to PostgreSQL.")
        except Exception as e:
            logging.error(f"Error writing to PostgreSQL in epoch {epoch_id}: {str(e)}", exc_info=True)
            # Don't raise here - allow the Redis write to still attempt

        # Write to Redis (Feast Online Store) 
        try:
            logging.info(f"Attempting to write to Redis: {CONFIG['redis']['host']}:{CONFIG['redis']['port']}")
            
            final_df.write \
                .format("org.apache.spark.sql.redis") \
                .option("table", "feast_online_store") \
                .option("key.column", "transaction_id") \
                .option("host", CONFIG["redis"]["host"]) \
                .option("port", CONFIG["redis"]["port"]) \
                .mode("append") \
                .save()
            logging.info(f"Epoch {epoch_id} - Successfully wrote to Redis.")
        except Exception as e:
            logging.error(f"Error writing to Redis in epoch {epoch_id}: {str(e)}", exc_info=True)

    except Exception as e:
        logging.error(f"Error in epoch {epoch_id}: {str(e)}", exc_info=True)
        try:
            logging.error("Showing sample data that caused the error:")
            batch_df.show(5, truncate=False)
        except:
            logging.error("Could not show sample data due to additional error")
        raise

def main():
    logging.info("Initializing Spark session")
    
    spark = SparkSession.builder \
    .appName("FraudDetectionStreaming") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.4,redis.clients:jedis:3.7.0") \
    .config("spark.jars", "/mnt/d/real_time_streaming/spark-redis_2.12-3.5.0.jar,/mnt/d/real_time_streaming/postgresql-42.6.2.jar") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .getOrCreate()

    # Set log level for Spark itself
    spark.sparkContext.setLogLevel("WARN")
    
    logging.info(f"Connecting to Kafka topic: {CONFIG['kafka']['topic']}")
    
    # Read from Kafka with extensive error handling
    try:
        # Raw Kafka Input - capture original message for debugging
        raw_df = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", CONFIG["kafka"]["broker"]) \
            .option("subscribe", CONFIG["kafka"]["topic"]) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .load()
        
        logging.info("Successfully connected to Kafka")
        
        # Add a debug stream to examine raw messages first
        raw_query = raw_df \
        .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
        .writeStream \
        .foreachBatch(lambda df, id: logging.info(f"Raw Kafka message batch {id}: {df.head(1)}")) \
        .trigger(processingTime="10 seconds") \
        .start()
        
        # Parse the stream - UPDATED to match the actual incoming JSON structure
        parsed_stream = raw_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select("data.*") \
            .withColumn("event_timestamp", unix_timestamp(col("trans_date_trans_time"), "yyyy-MM-dd HH:mm:ss").cast(TimestampType())) \
            .withColumn("trans_month", month(col("event_timestamp")).cast(IntegerType())) \
            .withColumn("trans_year", year(col("event_timestamp")).cast(IntegerType()))
        
        # CRITICAL FIX: Add debug query to verify the parsed data structure
        parsed_query = parsed_stream \
            .writeStream \
            .foreachBatch(lambda df, id: logging.info(f"Parsed data batch {id}: {df.limit(1).columns}")) \
            .trigger(processingTime="10 seconds") \
            .start()
        
        # Stream to PostgreSQL & Redis
        logging.info("Starting stream processing to downstream systems")
        final_query = parsed_stream.writeStream \
            .foreachBatch(write_to_stores) \
            .outputMode("append") \
            .option("checkpointLocation", CONFIG["checkpoint"]) \
            .trigger(processingTime="15 seconds") \
            .start()

        logging.info("All streaming queries started. Awaiting termination.")
        final_query.awaitTermination()
        
    except Exception as e:
        logging.error(f"Critical error in main process: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        logging.info("Starting main application")
        main()
    except Exception as e:
        logging.error(f"Application failed: {str(e)}", exc_info=True)