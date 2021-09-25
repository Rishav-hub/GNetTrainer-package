import base64


def decodeImage(imgstring, fileName):
    """Decode an image from a base64 string and write it to a file.

    Args:
        imgstring ([string]): [Image path to be decode]
        fileName ([string]): [Name of the file]
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):

    """Decode an image from a base64 string and write it to a file.

    Args:
        croppedImagePath ([string]): [Image path to be decode]
        
    """
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())