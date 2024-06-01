from .training_utils import print_example, print_training_update, print_2D

from datetime import datetime

def now(format="%H:%M:%S"):
    return datetime.now().strftime(format)

