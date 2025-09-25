import re
import sys
import logging

from tqdm import tqdm


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): The file path for log output. If None, logs to stdout.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Redirect tqdm's output to the correct stream
    tqdm.pandas(file=sys.stdout if not log_file else open(log_file, 'a'))

    return logger

def preprocess_text(text: str) -> str:
    """
    Performs preprocessing on a given text string.

    Args:
        text (str): The original text string.

    Returns:
        str: The cleaned and standardized text.
    """
    if not isinstance(text, str):
        return ""
    # Replace sequences of underscores
    text = re.sub(r'___+', '', text)
    # Remove non-alphanumeric characters except for basic punctuation
    text = re.sub(r'[^\w\s.,!?;]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove MIMIC de-identification placeholders
    text = text.replace('name unit no admission date discharge date date of birth ', '')
    return text