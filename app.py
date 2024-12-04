import cv2
import faceDetection as fD
import numpy as np
import cnn_models as cnn
import datasetHandelr as dH
print("ON")
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# label_encoder = LabelEncoder()
# checkpoint = ModelCheckpoint(
#     filepath='URPATH/MODELE/second_A.keras',  
#     monitor='val_loss',              
#     save_best_only=True,             
#     save_weights_only=False,         
#     mode='min',                      
#     verbose=1                        
# )




# images_path="URPATH/images"
# X,Y=dH.create_train_set(images_path)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=420)
# shape=(48, 48, 3)
# n_classes=7
# y_train = label_encoder.fit_transform(y_train) 

# y_train = to_categorical(y_train)
# X_train = np.array(X_train)
# print(np.shape(X_train))
# print(np.shape(y_train))

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     zoom_range=0.2,
#     shear_range=0.2,
#     fill_mode='nearest'
# )
# datagen.fit(X_train)
# # model=cnn.create_double_model_II(n_classes=n_classes,input_s=shape)
# load_model=tf.keras.models.load_model("URPATH/second.keras") 
# history=load_model.fit(X_train,y_train, epochs=15,  validation_split=0.2, verbose=1,callbacks=[checkpoint])
load_model=tf.keras.models.load_model("URPATH/second.keras") 
e_array=["angry","disgust","fear","happy","neutral","sad","suprise","unknown"]

cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    emotion =fD.get_emotion_from_img(img,load_model)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"W:{e_array[emotion[0]]}", (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    cv2.imshow('Klatka z kamery', img)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#
cap.release()
cv2.destroyAllWindows()