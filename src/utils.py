import logging
import os

def setup_logging(mode):
    logger = logging.getLogger(mode)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        file_handler = logging.FileHandler(f'{mode}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger