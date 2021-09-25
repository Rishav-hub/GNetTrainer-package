import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from GNetTrainer.utils import data_management as dm
import matplotlib.pyplot as plt

class Predict:
    def __init__(self, image_path):
        self.image_path = image_path

    def predict_image(self):

        # Load the model
        model_path = os.listdir('SAVED_MODEL')
        model_path = model_path[-1]
        print(model_path)
        print('Starting of Prediction')
        model = load_model('SAVED_MODEL/' + model_path)
        imagepath = self.image_path
        predict = dm.manage_input_data(imagepath)


        result = model.predict(predict)
        print("End of Prediction")
        results = np.argmax(result, axis=1)
        # print(results)
        train, valid = dm.get_datagen()
        classes = train.class_indices
        classes = dict((v, k) for k, v in classes.items())

        final_class = classes[results[0]] 
        print(f'The Image is classified as {final_class}')
        return [{ "image_class" : final_class}]

    

