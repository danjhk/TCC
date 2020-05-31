from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, MaxPooling3D, Flatten, Dropout, Reshape
import numpy as np

def get_model(input_shape, num_classes):
    model = Sequential([
        Conv3D(10, (9, 7, 3), input_shape=input_shape, activation='relu', padding='valid', name='conv_1'),
        MaxPooling3D((3,3,1), padding='valid', name='pool_1'),
        Conv3D(30, (7,7,3), activation='relu', padding='valid', name='conv_2'),
        MaxPooling3D((3,3,1), padding='valid', name='pool_2'),
        Reshape((6, 4, 150), name='reshape'),
        Conv2D(128, (6,4), padding='valid', activation='relu', name='conv_3'),
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
                        epochs=epochs, batch_size = batches)
    return history

def evaluate_acc(model, x_test, y_test, num_classes, stacks_per_list):
    prediction_list = np.zeros((num_classes,1))
    accuracy = 0
    for i in range(len(y_test)):
        x_test_f, y_test_f = preprocess(x_test.iloc[i],y_test.iloc[i],
                                    'Weizmann', stacks_per_list)
        model_prediction = model.predict(x_test_f)
        for stack in model_prediction:
            prediction_list[np.argmax(stack)] += 1
        if (np.argmax(prediction_list) == np.argmax(y_test.iloc[i])):
            accuracy += 1
            # print("Acerto")
        # print(prediction_list)
        # print(y_test.iloc[i])
        prediction_list = np.zeros((10,1))
    accuracy /= len(y_test)
    return accuracy;