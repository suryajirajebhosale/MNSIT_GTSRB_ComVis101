import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
from tensorflow.keras import Model

# TENSORFLOW.KERAS.SEQUENTIAL METHOD:

seq_model = tf.keras.Sequential(

[
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPool2D(),
    BatchNormalization(),
    GlobalAvgPool2D(),
    Dense(32, activation='relu'),
    Dense(10, activation = 'softmax')

    
]
)


# FUNCTIONAL MODEL

def functional_model():
    my_input = Input(shape=(28,28,1))
    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = Conv2D(64, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(10, activation = 'softmax')(x)
    model = tf.keras.Model(inputs = my_input, outputs = x)
    return model


def functional_model2(i,d):
    my_input = Input(shape=(28,28,1))
    x = Conv2D(i, (3,3), activation='relu')(my_input)
    x = Conv2D(i*2, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(d, activation='relu')(x)
    x = Dense(d/2, activation = 'softmax')(x)

    model_1 = tf.keras.Model(inputs = my_input, outputs = x)
    return model_1


# CLASS METHOD

class mycustommodel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.conv = Conv2D(32, (3,3), activation='relu')
        self.maxp = MaxPool2D()
        self.batchnorm = BatchNormalization()
        self.glopool = GlobalAvgPool2D()
        self.dense1 = Dense(32, activation='relu')
        self.dense2 = Dense(10, activation = 'softmax')


    def call(self, my_input):
        x = self.conv(my_input)
        x = self.maxp(x)
        x = self.batchnorm(x)
        x = self.glopool(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x



def sts_model(n_classes):
    my_input = Input(shape=(60,60,3))
    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = Conv2D(64, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(43, activation = 'softmax')(x)
    model = tf.keras.Model(inputs = my_input, outputs = x)
    return model


