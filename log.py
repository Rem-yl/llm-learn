import logging
from logging.handlers import RotatingFileHandler

# Configure the logger
logger = logging.getLogger("LLM_Learn")
logger.setLevel(logging.DEBUG)

# Create a file handler that logs debug and higher level messages
handler = RotatingFileHandler("llm.log", maxBytes=200000, backupCount=5)
handler.setLevel(logging.DEBUG)
# Create a console handler that logs warning and higher level messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Create a formatter and set it for both handlers
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d - %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(console_handler)
