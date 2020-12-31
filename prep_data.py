import os
import json 
import librosa
from PIL import Image
import cv2
import numpy as np 
from conf import DATA_DIR

DATA_SET = os.path.join(DATA_DIR,"BIKES")
os.makedirs(DATA_SET,exist_ok=True)


DATA_SET = os.path.join(DATA_DIR,"CARS")
os.makedirs(DATA_SET,exist_ok=True)


DATA_SET = 'img_data'
JSON_PATH = 'image_test_tst.json'
NEW_DATA_SET = 'DATA'


def PREPARE_DATASET(data_set,json_path,new_data_set):

    data = {
        'mappings' : [],
        'labels' : [],
        'files' : [],
        'PIXs' : [] 
        
    }





    for i , (dirpath , dirnames , filenames) in enumerate(os.walk(data_set)):
        if dirpath is not data_set:
            category = dirpath.split('/')[0]
            #data['mappings'].append(category)
            print(f'Processing : {category}')

            print(f'dirpath : {dirpath}')
            print(f'dirnames : {dirnames}')
            print(f'filenames : {filenames}')

            if dirpath == 'img_data\Bikes':
                for f in filenames:
                    file_path = os.path.join(dirpath,f)
                    bike_img = Image.open(file_path)
                    new_bike_img = bike_img.resize((100,100))
                    new_bike_img.save(f'DATA/BIKES/{f}')
                    print(f'old bike img size : {bike_img.size}')
                    print(f'new bike img size : {new_bike_img.size}')

            elif dirpath == 'img_data\Cars':
                for f in filenames:
                    file_path = os.path.join(dirpath,f)
                    car_img = Image.open(file_path)
                    new_car_img = car_img.resize((100,100))
                    new_car_img.save(f'DATA/CARS/{f}')
                    print(f'old car img size : {car_img.size}')
                    print(f'new car img size : {new_car_img.size}')


    for i , (dirpath , dirnames , filenames) in enumerate(os.walk(new_data_set)):

        if dirpath is not data_set:
            category = dirpath.split('/')[0]
            data['mappings'].append(category)
            print(f'Processing : {category}')

            print(f'dirpath : {dirpath}')
            print(f'dirnames : {dirnames}')
            print(f'filenames : {filenames}')


            for f in filenames:

                file_path = os.path.join(dirpath,f)
                print(f'file : {file_path}')


                #loading pixels
                img_PIXs =  cv2.imread(file_path)

                data['labels'].append(i-1)
                data['PIXs'].append(img_PIXs.tolist())
                data['files'].append(file_path)
                print(f'File_path: {i-1}')

            with open(JSON_PATH,'w') as fp:
                json.dump(data,fp,indent=4)



if __name__ == '__main__':
    PREPARE_DATASET(DATA_SET,JSON_PATH,NEW_DATA_SET)
