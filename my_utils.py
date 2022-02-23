from asyncore import read
import shutil
from tkinter import image_names
from unittest import expectedFailure
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import shutil, glob
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def display_examples(examples, labels):

    plt.figure(figsize=(10,9))
    
    for i in range(0,25):
        idx = np.random.randint(0, examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5,i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    
    plt.show()

def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):

    folders = os.listdir(path_to_data)
    folders.remove('.DS_Store')

    for folder in folders:

        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))

        x_train, x_val = train_test_split(images_paths, test_size=split_size)

        for x in x_train:

            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

        for x in x_val:

            path_to_folder = os.path.join(path_to_save_val, folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder) 

def order_tset(path_to_image, path_to_csv):

    try:
        with open(path_to_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter = ',')

            for i, row in enumerate(reader):
                if i == 0:
                    continue
                imgname = row[-1].replace('Test/', '')
                label = row[-2]

                path_to_folder = os.path.join(path_to_image, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                full_path = os.path.join(path_to_image, imgname)
                shutil.move(full_path, path_to_folder)

    except:
        print('[INFO]: Error reading CSV File')




def create_generators(batch_size, train_data_path, val_data_path, test_data_path):

    train_preprocessor = ImageDataGenerator(
        rescale = 1 / 255
    )

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = train_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    test_generator = train_preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    return train_generator, val_generator, test_generator


# if __name__ == '__main__':
#         path_to_train = '//Users//bhosle//Desktop//intro_to_tf//GTSRB//trainingdata//train'
#         path_to_validation = '//Users//bhosle//Desktop//intro_to_tf//GTSRB//trainingdata//val'
#         path_to_test = '//Users//bhosle//Desktop//intro_to_tf//GTSRB//Test'
#         create_generators(batch_size = 32, train_data_path = path_to_train, val_data_path = path_to_validation, test_data_path = path_to_test)
