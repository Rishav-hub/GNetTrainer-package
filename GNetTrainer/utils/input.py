from GNetTrainer.utils.config import process_config
import os
from GNetTrainer.utils.model_config import return_model
from GNetTrainer.utils.close_match import get_match as gm

config = process_config("config.yaml")

print("Entered input file")
## Data configs

TRAIN_DIR = config["train_dir"]
VAL_DIR = config["val_dir"]

CLASSES = len(os.listdir(TRAIN_DIR))
DIMENSION = config["dimension"].split(',')

IMAGE_SIZE = int(DIMENSION[0]), int(DIMENSION[1]), int(DIMENSION[2])
BATCH_SIZE = config["batch_size"]

AUGMENTATION = config["augmentation"]

# print(CLASSES, IMAGE_SIZE, BATCH_SIZE, AUGMENTATION)

# Model config
MODEL_OBJ = config['MODEL_OBJ']
MODEL_OBJ = return_model(gm(MODEL_OBJ))
MODEL_NAME = config['MODEL_NAME'] 
EPOCHS = config['EPOCHS'] 
OPTIMIZER = config['OPTIMIZER']
LOSS_FUNC = config['LOSS_FUNC']
FREEZE_ALL = config['FREEZE_ALL']
TENSORBOARD = config['TENSORBOARD'] # Tensorboard true or false

# print(MODEL_OBJ, MODEL_NAME, EPOCHS, OPTIMIZER, LOSS_FUNC, FREEZE_ALL, TENSORBOARD)
# LR Scheduler

SCHEDULER = config['SCHEDULER'] # Learning rate scheduler true or false
MONITOR = config['MONITOR'] # Monitor the loss or accuracy
PATIENCE = config['PATIENCE'] # Patience of the scheduler
FACTOR = config['FACTOR'] # Factor of the scheduler


# print(SCHEDULER, LR_SCHEDULER, MONITOR, PATIENCE, FACTOR)



def configureData(TRAIN_DIR = TRAIN_DIR, VAL_DIR = VAL_DIR, AUGMENTATION = AUGMENTATION, CLASSES = CLASSES, IMAGE_SIZE = IMAGE_SIZE, BATCH_SIZE = BATCH_SIZE):

    """Loads the input from the user for configuring the data
    Returns:
        dictionary: key value pairs
    """
    CONFIG = {
        'TRAIN_DIR' : TRAIN_DIR,
        'VAL_DIR' : VAL_DIR,
        'AUGMENTATION': AUGMENTATION,
        'CLASSES' : CLASSES,
        'IMAGE_SIZE' : IMAGE_SIZE,
        'BATCH_SIZE' : BATCH_SIZE,
    }

    return CONFIG

def configureModel(MODEL_OBJ = MODEL_OBJ, MODEL_NAME=MODEL_NAME, EPOCHS = EPOCHS, FREEZE_ALL= FREEZE_ALL , OPTIMIZER=OPTIMIZER, LOSS_FUNC=LOSS_FUNC, TENSORBOARD=TENSORBOARD):
    """Loads the input from the user for configuring the model
    Returns:
        dictionary: key value pairs
    """
    CONFIG = {
        'MODEL_OBJ' : MODEL_OBJ,
        'MODEL_NAME' : MODEL_NAME,
        'EPOCHS' : EPOCHS,
        'FREEZE_ALL' : FREEZE_ALL,
        'OPTIMIZER': OPTIMIZER,
        'LOSS_FUNC' : LOSS_FUNC,
        'TENSORBOARD': TENSORBOARD,
    }

    return CONFIG

def lrscheduler(SCHEDULER = SCHEDULER, MONITOR = MONITOR, PATIENCE = PATIENCE, FACTOR = FACTOR):

    """Loads the input from the user for configuring the LR Scheduler
    Returns:
        dictionary: key value pairs
    """

    CONFIG = {
        'SCHEDULER' : SCHEDULER,
        'MONITOR' : MONITOR,
        'PATIENCE' : PATIENCE,
        'FACTOR' : FACTOR,
    }

    return CONFIG