from pickletools import optimize
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
from deeplearning_models import functional_model, mycustommodel
from my_utils import display_examples 




if __name__ == '__main__':
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    
    print('Shape of X_train: ',X_train.shape)
    print('Shape of y_train: ',y_train.shape)
    print('Shape of X_test: ',X_test.shape)
    print('Shape of y_test: ',y_test.shape)


    
    if False:
        display_examples(X_train, y_train)

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = np.expand_dims(X_train, axis = -1)
    X_test = np.expand_dims(X_test, axis = -1)

    print('Shape of X_train: ',X_train.shape)
    print('Shape of y_train: ',y_train.shape)
    print('Shape of X_test: ',X_test.shape)
    print('Shape of y_test: ',y_test.shape)

 
    # model = functional_model()
    model = mycustommodel()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')

    # Model Training
    model.fit(X_train, y_train, batch_size=32, epochs=3, validation_split = 0.2)

    # Evaluation
    model.evaluate(X_test, y_test, batch_size = 32)



