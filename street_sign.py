from gc import callbacks
import os 
import glob
from tabnanny import verbose
from sklearn.model_selection import train_test_split
import shutil
from my_utils import split_data, order_tset, create_generators
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from deeplearning_models import sts_model
import tensorflow as tf
from tensorflow.keras import models
     


if __name__ == '__main__':
    if False:
        path_to_data = '//Users//bhosle//Desktop//intro_to_tf//GTSRB//Train'
        path_to_save_train = '//Users//bhosle//Desktop//intro_to_tf//GTSRB//trainingdata/train'
        path_to_save_validation = '//Users//bhosle//Desktop//intro_to_tf//GTSRB//trainingdata/val'
        
        split_data(path_to_data, path_to_save_train = path_to_save_train, path_to_save_val = path_to_save_validation)

    if False:
        path_to_image = '//Users//bhosle//Desktop//intro_to_tf//GTSRB//Test'
        path_to_csv = '//Users//bhosle//Desktop//intro_to_tf//GTSRB//Test.csv'
        order_tset(path_to_image=path_to_image, path_to_csv=path_to_csv)

    
    
    path_to_train = '//Users//bhosle//Desktop//intro_to_tf//GTSRB//trainingdata/train'
    path_to_validation = '//Users//bhosle//Desktop//intro_to_tf//GTSRB//trainingdata/val'
    path_to_test = '//Users//bhosle//Desktop//intro_to_tf//GTSRB//Test'
    batch_size = 64
    epochs = 2

    
    train_generator, validation_generator, test_generator = create_generators(batch_size, path_to_train, path_to_validation, path_to_test)
    n_classes = train_generator.num_classes

    Tra = False
    Tes = True
    
    if Tra:
        path_to_save_model = '//Users//bhosle//Desktop//intro_to_tf//models'
        check_saver = ModelCheckpoint(
            path_to_save_model,
            monitor= 'val_accuracy',
            mode='max',
            save_best_only=True
        )

        early_stop = EarlyStopping(
            monitor='val_accuracy', 
            patience=10
        )

        model = sts_model(n_classes=n_classes)
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')

        model.fit(train_generator, epochs=epochs, batch_size = batch_size, 
                    validation_data = validation_generator, callbacks = check_saver)

    if Tes:
        model = tf.keras.models.load_model('/Users/bhosle/Desktop/intro_to_tf/models')
        model.summary()
        print('Validation: ')
        model.evaluate(validation_generator)
        print('Test')
        model.evaluate(test_generator)
    