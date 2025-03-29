import requests
import json
from elasticsearch import Elasticsearch
from datetime import datetime
from src.logging.otel_logger import logger

# Initialize Elasticsearch client
es = Elasticsearch([{'host': '54.175.59.242', 'port': 9200}])

def send_alert_to_elasticsearch(invalid_data):

    # Get current timestamp
    timestamp = datetime.utcnow().isoformat()

    alert_message = {
        'timestamp': timestamp, 
        'status': 'failed validation',
        'invalid_data': invalid_data
    }

    try:
        # Index the invalid data as an alert in Elasticsearch
        es.index(index="alerts", document=alert_message)
        logger.info(f"Alert sent to Elasticsearch: {alert_message}")
    except Exception as e:
        logger.error(f"Failed to send alert to Elasticsearch: {str(e)}")

def send_alert_to_alertmanager(alert_message):
    # URL for Alertmanager API
    alertmanager_url = "http://54.175.59.242:9093/api/v1/alerts"  # Replace with the actual address of Alertmanager
    
    # Prepare the alert data
    alert_data = [
        {
            "labels": {
                "alertname": "ValidationFailed",  # Custom label for your alert
                "severity": "critical"
            },
            "annotations": {
                "summary": alert_message
            },
            "startsAt": datetime.utcnow().isoformat()  # Timestamp of the alert
        }
    ]
    
    try:
        # Send the alert to Alertmanager via the HTTP API
        response = requests.post(alertmanager_url, data=json.dumps(alert_data), headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            logger.info("Alert successfully sent to Alertmanager!")
        else:
            logger.error(f"Failed to send alert to Alertmanager. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logger.error(f"Error sending alert to Alertmanager: {str(e)}")



# import os
# import requests
# import json
# from src.logging.otel_logger import logger  

# SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")  # Ensure this is set in your environment

# def send_slack_alert(message: str, channel: str = "#general"):
#     if not SLACK_WEBHOOK_URL:
#         raise ValueError("SLACK_WEBHOOK_URL is not set in environment variables.")

#     payload = {
#         "channel": channel,
#         "text": message
#     }
#     headers = {"Content-Type": "application/json"}
    
#     try:
#         response = requests.post(SLACK_WEBHOOK_URL, data=json.dumps(payload), headers=headers)
#         response.raise_for_status()  # Raise an exception for HTTP error responses
        
#         # Log successful Slack alert
#         logger.info(f"Slack alert sent successfully to {channel}: {message}")

#     except requests.exceptions.RequestException as e:
#         # Log error
#         logger.error(f"Failed to send Slack alert: {e}")
#         raise


# import requests
# import json
# from datetime import datetime

# class SlackAlertManager:
#     def __init__(self, webhook_url):
#         self.webhook_url = webhook_url

#     def send_alert(self, message, severity='warning'):
#         """
#         Send an alert to Slack
        
#         :param message: Alert message to send
#         :param severity: Alert severity (warning, critical, etc.)
#         """
#         payload = {
#             "text": f"*{severity.upper()} ALERT*: {message}",
#             "blocks": [
#                 {
#                     "type": "section",
#                     "text": {
#                         "type": "mrkdwn",
#                         "text": f"🚨 *{severity.upper()} ALERT*\n{message}"
#                     }
#                 }
#             ]
#         }

#         try:
#             response = requests.post(
#                 self.webhook_url, 
#                 data=json.dumps(payload),
#                 headers={'Content-Type': 'application/json'}
#             )
#             response.raise_for_status()
#         except Exception as e:
#             print(f"Failed to send Slack alert: {e}")

# # Create a global instance with your Slack webhook
# SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/YOUR_WEBHOOK_PATH'
# slack_alert = SlackAlertManager(SLACK_WEBHOOK_URL)