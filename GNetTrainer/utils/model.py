import os
import tensorflow as tf
from GNetTrainer.utils.input import configureData, configureModel, lrscheduler
import time

config_data = configureData()
config_model = configureModel()
config_scheduler = lrscheduler()

def get_model():
    """Check if the model is pretrained or not.

    Returns:
        model: Returns the model object.
    """    

    try:
        model = config_model['MODEL_OBJ']   
        print('Model already exists')
        return model
    except Exception as e:
        print("Model does not exist's")


def model_compile(model):
    """Compile the model.

    Args:
        model (object): model object

    Returns:
        object: model object
    """    
    print("Staring Model Preparation")

    if config_model['FREEZE_ALL'] == True:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in model.layers:
            layer.trainable = True
    
    
    # add custom layers -
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(
        units=config_data['CLASSES'],
        activation="softmax"
    )(flatten_in)

    full_model = tf.keras.models.Model(
        inputs=model.input,
        outputs = prediction
    )
    print("custom model summary")
    full_model.summary()



    full_model.compile(
        optimizer = config_model['OPTIMIZER'],
        loss = config_model['LOSS_FUNC'],
        metrics = ["accuracy"]
    )

    print("Model Preparation Complete")
    return full_model

def load_pretrain_model():

    """The logic for loading pretrain model.
  
     Args:
      MODEL_OBJ: Model object
    Returns:
      It returns keras model objcet
    """
    model = get_model()
    model = model_compile(model)
    return model

def callbacks(base_dir="."):
    """ The logic for callbacks.
    Args: 
    """
    print("Setting Up Callbacks")

    ## Tensorboard callback
    TENSOR_DIR = "tensorboard_log_dir"
    base_log_dir = os.makedirs(TENSOR_DIR, exist_ok=True)
    unique_log = time.strftime(r"log_at_%Y%m%d_%H%M%S")
    tensorboard_log_dir = os.path.join(TENSOR_DIR, unique_log)
    print(base_log_dir)
    # tensorboard_log_dir = base_log_dir + "/" + str(unique_log)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)

    print("Tensorboard Callbacks Initiated")

    # Checkpoint callback
    CHECK_DIR = "checkpoint_dir"
    base_checkpoint_dir = os.makedirs(CHECK_DIR, exist_ok=True)
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CHECK_DIR, save_best_only=True,
                                         monitor=config_scheduler['MONITOR'])
    callback_list = [tensorboard_cb, checkpointing_cb]

    if config_scheduler['SCHEDULER'] == True:
        print("Using Lr Scheduler")
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor = config_scheduler['MONITOR'],
                                        factor=config_scheduler['FACTOR'],
                                            patience=config_scheduler['PATIENCE'])
        callback_list = [tensorboard_cb, checkpointing_cb, lr_scheduler]

    print("Callbacks Initiated")

    return callback_list

def saveModel_path(model_dir="."):
    """The logic for saving model path.
    """
    model_dir = 'SAVED_MODEL'
    os.makedirs(model_dir, exist_ok=True)
    model_name = config_model['MODEL_NAME']
    fileName = time.strftime(f"{model_name}_%Y_%m_%d_%H_%M_%S_.h5")    
    model_path = os.path.join(model_dir, fileName)
    print(f"Your model will be saved at the following location\n{model_path}")
    return model_path




# print(load_pretrain_model())