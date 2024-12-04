import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,Input,LeakyReLU,GlobalAveragePooling2D, Conv1D,MaxPooling1D, Softmax
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras import regularizers

def create_model_II(input_s, n_classes):
    model = Sequential()
    model.add(Input(shape=input_s))
    model.add(Conv2D(64, (4, 4), activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (4, 4), activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3)) 
    model.add(Conv2D(128, (3,3), activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(0.4))   
    model.add(Dense(n_classes, activation='softmax'))

    adamOpti = Adam(learning_rate=0.0001)
    model.compile(optimizer=adamOpti, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_test(input_s, n_classes):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_s))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


#best model
def create_double_model_II(input_s, n_classes):

    model = Sequential()
    model.add(Input(shape=input_s))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), activation='elu'))
    model.add(Conv2D(128, (3, 3), activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (3, 3), activation='elu'))
    model.add(Conv2D(256, (3, 3), activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256, activation='elu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))   
    model.add(Dense(n_classes, activation='softmax'))

    adamOpti = Adam(learning_rate=0.001)
    model.compile(optimizer=adamOpti, loss='categorical_crossentropy', metrics=['accuracy'])
    return model