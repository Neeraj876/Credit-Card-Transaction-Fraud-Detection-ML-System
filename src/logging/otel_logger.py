import logging
import os
import sys
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.log_exporter import OTLPLogExporter
from opentelemetry.sdk.logs import LoggerProvider, LoggingHandler
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader 

# Load OTLP endpoint from environment variables (better for production)
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", "http://otel-collector:4317")

# Define OpenTelemetry Resource
resource = Resource.create({"service.name": "my-app", "env": "production"})

# Configure Tracing
tracer_provider = TracerProvider(resource=resource)
trace_exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# Configure Logging
logger_provider = LoggerProvider(resource=resource)
log_exporter = OTLPLogExporter(endpoint=OTLP_ENDPOINT)
logging_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)

# Setup Python Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging_handler,  # Sends logs to OpenTelemetry
        logging.StreamHandler(sys.stdout),  # Console output
    ],
)

logging = logging.getLogger("otel-logger")

# Configure OpenTelemetry Metrics
metric_exporter = OTLPMetricExporter(endpoint=OTLP_ENDPOINT)
metric_reader = PeriodicExportingMetricReader(metric_exporter)
meter_provider = MeterProvider(metric_readers=[metric_reader])
set_meter_provider(meter_provider)

# Example Usage
if __name__ == "__main__":
    logging.info("Application started successfully!")
    logging.warning("This is a warning message.")
    logging.error("An error occurred in the system.")

    # Example tracing
    with tracer.start_as_current_span("example-operation"):
        logging.info("Tracing an example operation.")

# import logging
# import os
# import sys
# from opentelemetry import trace
# from opentelemetry.sdk.resources import Resource
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
# from opentelemetry.sdk.logs import LoggerProvider, LoggingHandler

# # Load OTLP endpoint from environment variables (better for production)
# OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", "http://otel-collector:4317")

# # Define OpenTelemetry Resource
# resource = Resource.create({"service.name": "my-app", "env": "production"})

# # Configure Tracing
# tracer_provider = TracerProvider(resource=resource)
# trace_exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
# tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
# trace.set_tracer_provider(tracer_provider)
# tracer = trace.get_tracer(__name__)

# # Configure Logging
# logger_provider = LoggerProvider(resource=resource)
# log_exporter = OTLPLogExporter(endpoint=OTLP_ENDPOINT)
# logging_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)

# # Setup Python Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging_handler,  # Sends logs to OpenTelemetry
#         logging.StreamHandler(sys.stdout),  # Console output
#     ],
# )

# logger = logging.getLogger("otel-logger")

# # Example Usage
# if __name__ == "__main__":
#     logger.info("Application started successfully!")
#     logger.warning("This is a warning message.")
#     logger.error("An error occurred in the system.")
