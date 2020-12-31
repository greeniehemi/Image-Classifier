import tensorflow.keras  as keras
import numpy as np
import cv2
from PIL import Image
#import librosa

MODEL_PATH = "test_model_2.h5"
height , width = 100,100

class _vehicle_spotting_service:
    model = None
    _mappings = [
        'Bikes',
        'Cars'
        ]

    _instance = None

    def predict(self,file_path):

        #extract MFCCs
        PIXs = self.process(file_path) # (#segments,#coefficients)

        #convert 2d MFCCs array into 4d array(samples,#segments,#coefficients,channel)
        PIXs = PIXs[np.newaxis,...]

        #make prediction
        predictions = self.model.predict(PIXs)
        predictions_index = np.argmax(predictions)
        predictions_words = self._mappings[predictions_index]
        return predictions_words


    def process(self,file_path):
        #load audio file
        img = Image.open(file_path)

        #setting the size of audio file
        new_img = img.resize((height,width))
        new_img.save('new img/testimg.jpg')

        

        # getting the MFCCs
        PIXs = cv2.imread('new img/testimg.jpg')
        PIXs= PIXs
        return PIXs


def Vehicle_Spotting_Service():

    #ensurre  that we have only one instance of KSS
    if _vehicle_spotting_service._instance is None:
        _vehicle_spotting_service._instance = _vehicle_spotting_service()
        _vehicle_spotting_service.model = keras.models.load_model(MODEL_PATH)
        return _vehicle_spotting_service._instance


if __name__ == '__main__':

    VSS = Vehicle_Spotting_Service()
    
    print('Please enter the path of your Image to be classified')
    image_input = input()

    image_input = str(image_input)
    Keyword1 = VSS.predict(f'{image_input}.jpg')
    #Keyword2 = KSS.predict('')
    print()
    print(f"Predicted Vehicle is :{Keyword1}")
