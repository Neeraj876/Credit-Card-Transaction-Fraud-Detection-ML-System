import os
import sys
import json

from dotenv import load_dotenv
import certifi
import pandas as pd
import pymongo
from src.exception.exception import CreditCardException
from src.logging.logger import logging

load_dotenv()

# Get MongoDB URL from environment variables
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
if not MONGO_DB_URL:
    logging.error("MONGO_DB_URL is not set in the environment variables.")
    raise ValueError("MONGO_DB_URL is not set in the environment variables.")

# Certificate for MongoDB connection
ca = certifi.where()

class DataExtract:
    def __init__(self):
        try:
            logging.info("DataExtract object initialized successfully.")
        except Exception as e:
            logging.error("Error initializing DataExtract object.")
            raise CreditCardException(e, sys)

    def csv_to_json_convertor(self, file_path, chunk_size=10000):
        """
        Reads a CSV file in chunks, converts it to JSON, and returns a list of records.
        """
        try:
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")

            logging.info(f"Reading CSV file in chunks from {file_path}...")
            records = []

            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunk.reset_index(drop=True, inplace=True)
                chunk_records = json.loads(chunk.to_json(orient="records"))
                records.extend(chunk_records)

            logging.info(f"Successfully converted data to JSON format. Total records: {len(records)}")

            return records
        except Exception as e:
            logging.error("Error in CSV to JSON conversion", exc_info=True)
            raise CreditCardException(e, sys)

    def insert_data_mongodb(self, records, database, collection, batch_size=1000):
        """
        Inserts records into MongoDB in batches.
        """
        try:
            logging.info(f"Connecting to MongoDB database: {database}, collection: {collection}")
            mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            db = mongo_client[database]
            coll = db[collection]

            logging.info("Inserting data into MongoDB in batches...")

            for i in range(0, len(records), batch_size):
                coll.insert_many(records[i:i + batch_size])  # Insert batch
                logging.info(f"Inserted {len(records[i:i + batch_size])} records so far...")

            logging.info(f"Successfully inserted {len(records)} records into MongoDB.")

            return len(records)
        except Exception as e:
            logging.error("Error inserting data into MongoDB", exc_info=True)
            raise CreditCardException(e, sys)

if __name__ == "__main__":
    try:
        FILE_PATH = "Notebooks/Dataset/creditCard.csv"
        database = "FRAUD"
        collection = "creditcardData"

        logging.info("Starting ETL process...")
        obj = DataExtract()
        
        records = obj.csv_to_json_convertor(FILE_PATH)
        no_of_records = obj.insert_data_mongodb(records, database, collection)

        logging.info(f"ETL process completed successfully. Total records inserted: {no_of_records}")
        print(f"Inserted {no_of_records} records into MongoDB.")
    
    except Exception as e:
        raise CreditCardException(e, sys)
    
    