from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, MaxPooling3D, Flatten, Dropout, Reshape
import numpy as np

def get_model(input_shape, num_classes):
    model = Sequential([
        Conv3D(10, (9, 7, 3), input_shape=input_shape, activation='tanh', padding='valid', name='conv_1'),
        MaxPooling3D((3,3,1), padding='valid', name='pool_1'),
        Conv3D(30, (7,7,3), activation='tanh', padding='valid', name='conv_2'),
        MaxPooling3D((3,3,1), padding='valid', name='pool_2'),
        Reshape((6, 4, 150), name='reshape'),
        Conv2D(128, (6,4), padding='valid', activation='tanh', name='conv_3'),
        Flatten(),
        Dense(num_classes, activation='softmax', name='dense-softmax')
    ])
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_network(model, X, y, epochs, batches):
    history = model.fit(X,
                        y,
                        verbose=1,
                        epochs=1)
    return history