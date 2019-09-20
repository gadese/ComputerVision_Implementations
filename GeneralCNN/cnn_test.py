from __future__ import print_function
import tensorflow as tf
#https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist#2.-How-many-feature-maps?
import plaidml.keras
plaidml.keras.install_backend()

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import os

batch_size = 32
num_classes = 10
epochs = 25
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')

# input image dimensions
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# The data, split between train and test sets:
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# models = [0] * 3
#
# model_name = 'mnist_convolutions'
# ##Testing number of convolutions to do
# for i in range(3):
#
#     models[i] = Sequential()
#     models[i].add(
#         Conv2D(filters=24, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape))
#     models[i].add(MaxPooling2D(pool_size=(2, 2), strides=2))
#
#     if i > 0:
#         models[i].add(Conv2D(filters=48, kernel_size=5, strides=1, padding='same', activation='relu'))
#         models[i].add(MaxPooling2D(pool_size=(2, 2), strides=2))
#
#     if i > 1:
#         models[i].add(Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu'))
#         models[i].add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
#
#     models[i].add(Flatten())
#     models[i].add(Dense(units=256, activation='relu'))
#     models[i].add(Dense(units=10, activation='softmax'))
#
#     models[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# #Testing how many feature maps
# models = [0] * 6
# nbr_feats = [8, 16, 24, 32, 48, 64]
# for i in range(0, len(models)):
#
#     models[i] = Sequential()
#     models[i].add(Conv2D(filters=nbr_feats[i], kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape))
#     models[i].add(MaxPooling2D(pool_size=(2, 2), strides=2))
#
#     models[i].add(Conv2D(filters=2*nbr_feats[i], kernel_size=5, strides=1, padding='same', activation='relu'))
#     models[i].add(MaxPooling2D(pool_size=(2, 2), strides=2))
#
#     models[i].add(Flatten())
#     models[i].add(Dense(units=256, activation='relu'))
#     models[i].add(Dense(units=10, activation='softmax'))
#
#     models[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# model_name = 'mnist_featuremaps'


# #Testing how large the dense layer is
# models = [0] * 7
# layer_units = [32, 64, 128, 256, 512, 1024, 2048]
# for i in range(0, len(models)):
#
#     models[i] = Sequential()
#     models[i].add(Conv2D(filters=48, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape))
#     models[i].add(MaxPooling2D(pool_size=(2, 2), strides=2))
#
#     models[i].add(Conv2D(filters=96, kernel_size=5, strides=1, padding='same', activation='relu'))
#     models[i].add(MaxPooling2D(pool_size=(2, 2), strides=2))
#
#     models[i].add(Flatten())
#     models[i].add(Dense(units=layer_units[i], activation='relu'))
#     models[i].add(Dense(units=10, activation='softmax'))
#
#     models[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# model_name = 'mnist_nbrunits'


#Testing how much dropout
models = [0] * 8
for i in range(0, len(models)):

    models[i] = Sequential()
    models[i].add(Conv2D(filters=48, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape))
    models[i].add(MaxPooling2D(pool_size=(2, 2), strides=2))
    models[i].add(Dropout(0.1*i))

    models[i].add(Conv2D(filters=96, kernel_size=5, strides=1, padding='same', activation='relu'))
    models[i].add(MaxPooling2D(pool_size=(2, 2), strides=2))
    models[i].add(Dropout(0.1 * i))

    models[i].add(Flatten())
    models[i].add(Dense(units=256, activation='relu'))
    models[i].add(Dropout(0.1 * i))

    models[i].add(Dense(units=10, activation='softmax'))

    models[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_name = 'mnist_dropout'

model_nbr = 0
histories = []
for my_model in models:

    if not data_augmentation:
        print('Not using data augmentation.')
        history=my_model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(x_test, y_test),
                     shuffle=True,
                     verbose=1)

    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Fit the model on the batches generated by datagen.flow().
        history=my_model.fit_generator(datagen.flow(x_train, y_train,
                                            batch_size=batch_size),
                               epochs=epochs,
                               validation_data=(x_test, y_test),
                               verbose=1,
                               workers=4,
                               steps_per_epoch=50000 / batch_size)

    histories.append(history)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name + str(model_nbr) + '.h5')
    my_model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = my_model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    model_nbr += 1

import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys() for history in histories)


fig, ax = plt.subplots(2, 1)
for history in histories:
    ax[0].plot(history.history['acc'])
    ax[1].plot(history.history['val_acc'])

ax[0].set_xlim(0, epochs-1)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('accuracy')
ax[0].set_title('model accuracy - Train')
ax[0].legend(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], loc='upper left')

ax[1].set_title('model accuracy - Test')
ax[1].set_xlim(0, epochs-1)
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].legend(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], loc='upper left')

fig.tight_layout()
plt.savefig('saved_models/'+model_name, format='png')
plt.show()
