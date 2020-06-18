import os

class Config():

    PROJECT_DIR = os.environ["PWD"]
    DATA_DIR = os.getenv('DATA_DIR', 'data/')
    RESULTS_DIR = os.getenv('RESULTS_DIR', 'results/')
    MODELS_DIR = os.getenv("MODELS_DIR", 'models/')
    LOGS_DIR = os.getenv("LOGS_DIR", 'logs/')