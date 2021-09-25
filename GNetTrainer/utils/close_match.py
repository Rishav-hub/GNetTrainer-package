import difflib

models = ["Xception", "VGG16", "VGG19", "ResNet50", "InceptionV3", "InceptionResNetV2", "MobileNet", "MobileNetV2", "DenseNet121", "DenseNet169", "DenseNet201", "NASNetMobile", "NASNetLarge"]
def get_match(input):
    """Gives the closest match to the input string

    Args:
        input (str): The input string whose closest match is to be found

    Returns:
        str: Returns the closest match to the input string
    """
    match = difflib.get_close_matches(input, models)
    return match[0]
    