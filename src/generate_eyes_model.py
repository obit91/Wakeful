import csv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation, Dropout, MaxPooling2D
from keras.optimizers import Adam
import tensorflow

HEIGHT = 26
WIDTH = 34


def readCsv(path):
    with open(path, 'r') as f:
        # read the scv file with the dictionary format
        reader = csv.DictReader(f)
        rows = list(reader)
        # imgs is a numpy array with all the images
        # tgs is a numpy array with the tags of the images

        imgs = np.empty((len(list(rows)), HEIGHT, WIDTH, 1), dtype=np.uint8)
        tgs = np.empty((len(list(rows)), 1))

        for row, i in zip(rows, range(len(rows))):

            # convert the list back to the image format
            img = row['image']
            img = img.strip('[').strip(']').split(', ')
            im = np.array(img, dtype=np.uint8)
            im = im.reshape((26, 34))
            im = np.expand_dims(im, axis=2)
            imgs[i] = im
            # the tag for open is 1 and for close is 0
            tag = row['state']
            if tag == 'open':
                tgs[i] = 1
            else:
                tgs[i] = 0

        # shuffle the dataset
        index = np.random.permutation(imgs.shape[0])
        imgs = imgs[index]
        tgs = tgs[index]

        # return images and their respective tags
        return imgs, tgs


#make the convolution neural network
def makeModel():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding = 'same', input_shape=(HEIGHT, WIDTH,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (2,2), padding= 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (2,2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train():
    x_train, y_train = readCsv('../res/dataset.csv')

    # scale the values of the images between 0 and 1
    x_train = x_train.astype('float32')
    x_train /= 255

    model = makeModel()
    
    # do some data augmentation
    aughmented_data = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )
    aughmented_data.fit(x_train)

    # train the model
    model.fit_generator(aughmented_data.flow(x_train, y_train, batch_size=32),
                        steps_per_epoch=len(x_train) / 32, epochs=50)

    # save the model
    model.save('../trained_models/trained_left_eye_detector.hdf5')


train()
