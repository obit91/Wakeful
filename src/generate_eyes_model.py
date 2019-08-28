import csv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation, Dropout, MaxPooling2D
from keras.optimizers import Adam
from matplotlib import pyplot as plt

HEIGHT = 26
WIDTH = 34
TRAINED_PATH = '../trained_models/trained_left_eye_detector.hdf5'

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


# make the convolution neural network
def generate_model():
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
    x, y = readCsv('../res/dataset.csv')

    # scale the values of the images between 0 and 1
    x = x.astype('float32')
    x /= 255

    x_train = x[0:2500]
    x_test = x[2500:]

    y_train = y[0:2500]
    y_test = y[2500:]

    model = generate_model()
    
    # do some data augmentation
    augmented_data = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        data_format='channels_last'
    )
    # aughmented_data.fit(x_train)

    batch_size = 40
    epochs = 50

    # train the model
    history = model.fit_generator(augmented_data.flow(x_train, y_train, batch_size=batch_size),
                                   steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                                   epochs=epochs,
                                   validation_data=(x_test, y_test),
                                   workers=4)

    # save the model
    model.save(TRAINED_PATH)

    model.evaluate(x_test, y_test)

    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)

    plt.show()


def load_trained_model():
    """
    Loads and returns the trained closed/open eye classification model.
    :return: A trained closed/open eye classification model.
    """
    model = generate_model()
    model.load_weights(TRAINED_PATH)
    return model


# train()
