import tensorflow as tf
import GNetTrainer.utils.data_management as dm
from GNetTrainer.utils import input, model
import time
configmodel = input.configureModel()
def train():
    

    my_model = model.load_pretrain_model()
    callbacks = model.callbacks()
    train_generator, valid_generator = dm.get_datagen()

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    try:
        my_model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=configmodel["EPOCHS"],
            steps_per_epoch=steps_per_epoch, 
            validation_steps=validation_steps,
            callbacks=callbacks
        )
    except Exception as e:
        print(f"Training stopped Due to {e}")

    print("#"*10)
    print("Training finished")
    print('#'*10)
    print("Saving model...")
    my_model.save(model.saveModel_path())





