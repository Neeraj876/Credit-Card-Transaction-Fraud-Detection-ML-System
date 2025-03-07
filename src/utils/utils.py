from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, MinMaxScaler
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, mean, stddev

import sys
from src.logging.logger import logging


def get_missing_columns(df, columns):
    """Returns a list of columns that contain missing values."""
    return [c for c in columns if df.filter(col(c).isNull()).count() > 0]


def get_non_constant_columns(df, columns):
    """Returns a list of numerical columns that have variance (to prevent scaling issues)."""
    return [c for c in columns if df.select(stddev(col(c))).collect()[0][0] is not None]


def create_pipeline(df, num_features, cat_features):
    stages = []

    # 1️⃣ Check and Handle Missing Values (Numerical & Categorical)
    num_missing = get_missing_columns(df, num_features)
    cat_missing = get_missing_columns(df, cat_features)

    if num_missing:
        num_imputer = Imputer(
            inputCols=num_missing,
            outputCols=[f"{c}_imputed" for c in num_missing],
            strategy="mean"
        )
        stages.append(num_imputer)
    else:
        print("✅ No missing numerical values detected, skipping imputation.")

    if cat_missing:
        cat_imputer = Imputer(
            inputCols=cat_missing,
            outputCols=[f"{c}_imputed" for c in cat_missing],
            strategy="mode"
        )
        stages.append(cat_imputer)
    else:
        print("✅ No missing categorical values detected, skipping imputation.")

    # 2️⃣ Ensure Correct Feature Assembly
    num_imputed = [f"{c}_imputed" if c in num_missing else c for c in num_features]
    cat_imputed = [f"{c}_imputed" if c in cat_missing else c for c in cat_features]

    num_assembler = VectorAssembler(
        inputCols=num_imputed,
        outputCol="num_vector"
    )
    stages.append(num_assembler)

    # 3️⃣ Handle MinMaxScaler Properly
    num_with_variance = get_non_constant_columns(df, num_features)
    if num_with_variance:
        num_scaler = MinMaxScaler(
            inputCol="num_vector",
            outputCol="num_scaled"
        )
        stages.append(num_scaler)
    else:
        print("⚠️ No variance in numerical columns, skipping MinMaxScaler.")

    # 4️⃣ Convert Categorical Features to Indexed Values
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep")
        for c in cat_imputed
    ]
    stages += indexers

    # 5️⃣ Final Feature Assembly
    final_assembler = VectorAssembler(
        inputCols=["num_scaled"] + [f"{c}_indexed" for c in cat_imputed],
        outputCol="features"
    )
    stages.append(final_assembler)

    return Pipeline(stages=stages)


def feature_engineering(df: DataFrame) -> DataFrame:
    """
    Performs feature engineering on the dataset by adding new features and removing unnecessary columns.

    Args:
        df (DataFrame): Input Spark DataFrame.

    Returns:
        DataFrame: Transformed DataFrame with new features.
    """
    try:
        # Add unique identifier for each transaction
        df = df.withColumn("transaction_id", F.monotonically_increasing_id())

        # Convert timestamp columns
        df = df.withColumn("trans_date_trans_time", F.to_timestamp("trans_date_trans_time"))
        df = df.withColumn("event_timestamp", F.col("trans_date_trans_time"))

        # Extract date components
        df = df.withColumn("trans_month", F.month("trans_date_trans_time"))
        df = df.withColumn("trans_year", F.year("trans_date_trans_time"))

        # Drop unwanted columns (only if they exist)
        columns_to_remove = [
            "_id", "Unnamed_0", "trans_date_trans_time", "first", "last",
            "street", "city", "state", "zip", "lat", "long", "dob",
            "trans_num", "unix_time", "merch_lat", "merch_long"
        ]
        existing_columns = list(set(df.columns) & set(columns_to_remove))
        df = df.drop(*existing_columns)

        return df

    except Exception as e:
        logging.error(f"❌ Error in feature engineering: {e}")
        raise Exception(f"Feature engineering error: {e}")
