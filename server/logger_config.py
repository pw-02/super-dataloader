import logging

def configure_logger():
    # Set the log levels for specific loggers to WARNING
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    
    # Configure the root logger with DEBUG level and a simple format
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    # Create and configure a logger named "SUPER"
    logger = logging.getLogger("SUPER")
    return logger

logger = configure_logger()
