import cv2
import os
import numpy as np
import pandas as pd

def create_train_set(dataset_path):
    train_path=dataset_path+"/train"
    X_train_Set=[]
    Y_train_Set=[]
    for root, dirs, files in os.walk(train_path):
        for file in files:
            
            file_path = os.path.join(root, file)
            image_path = file_path
            image = cv2.imread(image_path)
            X_train_Set.append(image)
            Y_train_Set.append(root.split("\\")[1])

    Y_train_Set=pd.DataFrame(Y_train_Set)
    return X_train_Set, Y_train_Set
def create_val_set(dataset_path):
    validation_path=dataset_path+'/validation'
