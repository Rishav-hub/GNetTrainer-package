from GNetTrainer.utils.input import configureData
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np

config = configureData()

def get_datagen():

    """This function generates train & valid data from images path,
    it also performs data augmentation.

    Returns:
        data: object of data
    """    

    if config['AUGMENTATION'] == True:

        print('Applying augmentation')
        train_datagen = ImageDataGenerator(rescale = 1./255, 
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip = True)
        valid_datagen = ImageDataGenerator(rescale = 1./255)

        training_set = train_datagen.flow_from_directory(config['TRAIN_DIR'], 
                                                 target_size = config['IMAGE_SIZE'][:-1],
                                                 batch_size = config['BATCH_SIZE'],
                                                 class_mode = 'categorical'
                                                 )

        valid_set = valid_datagen.flow_from_directory(config['VAL_DIR'], 
                                                 target_size = config['IMAGE_SIZE'][:-1],
                                                 batch_size = config['BATCH_SIZE'],
                                                 class_mode = 'categorical'
                                                 )
    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        valid_datagen = ImageDataGenerator(rescale = 1./255)

        training_set = train_datagen.flow_from_directory(config['TRAIN_DIR'], 
                                                 target_size = config['IMAGE_SIZE'][:-1],
                                                 batch_size = config['BATCH_SIZE'],
                                                 class_mode = 'categorical'
                                                 )

        valid_set = valid_datagen.flow_from_directory(config['VAL_DIR'], 
                                                 target_size = config['IMAGE_SIZE'][:-1],
                                                 batch_size = config['BATCH_SIZE'],
                                                 class_mode = 'categorical'
                                                 )


    print('Augmentation Process Done')
    return training_set, valid_set



def manage_input_data(input_image):
    """converting the input array into desired dimension
    Args:
        input_image (nd array): image nd array
    Returns:
        nd array: resized and updated dim image
    """
    try:

        images = input_image
        size = config['IMAGE_SIZE'][:-1]
        test_image = image.load_img(images, target_size = size)
        test_image = image.img_to_array(test_image)
        final_img = np.expand_dims(test_image, axis=0)

        return final_img

    except  Exception as e:
        print(e)
        print('Error in converting the input image')

    


# print(get_datagen())