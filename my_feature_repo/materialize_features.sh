#!/bin/bash

echo "Starting materialization at $(date)"

# Use the full path to activate
source /mnt/d/real_time_streaming/venv/bin/activate

# Navigate to your feature repository
cd /mnt/d/real_time_streaming/my_feature_repo/feature_repo

# Calculate timestamps for 30 minutes ago and now
END_DATE=$(date -u +"%Y-%m-%dT%H:%M:%S")
START_DATE=$(date -u -d "30 minutes ago" +"%Y-%m-%dT%H:%M:%S")

# Use full path to feast with time range
/mnt/d/real_time_streaming/venv/bin/feast materialize $START_DATE $END_DATE

echo "Finished materialization at $(date)"