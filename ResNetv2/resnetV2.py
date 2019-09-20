from __future__ import print_function
#import plaidml.keras
#plaidml.keras.install_backend()

import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, Input, Add, AveragePooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

from keras.utils import plot_model

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 2
data_augmentation = True
num_classes = 10
subtract_pixel_mean = True
nbr_stacks = 3
nbr_blocks = 6



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def conv_layer(inputs, nbr_filt= 16, kernel_size=3, strides=(1, 1), activ='relu', batch_norm=True, conv=True):
    convol = Conv2D(nbr_filt, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(1e-4))
    result= inputs

    if conv:
        result = convol(result)
        if batch_norm:
            result = BatchNormalization()(result)
        if activ is not None:
            result = Activation(activation=activ)(result)
    else:
        if batch_norm:
            result = BatchNormalization()(result)
        if activ is not None:
            result = Activation(activation=activ)(result)
        result = convol(result)

    return result

def resnetV2(shape, depth, nbr_classes):
    num_filters_in = 16
    nbr_blocks = int((depth - 2) / 9)

    inputs = Input(shape)
    data = conv_layer(inputs, nbr_filt=num_filters_in)

    for stack in range(nbr_stacks):
        for block in range(nbr_blocks):
            activation = 'relu'
            batch_normalization = True
            stride = (1, 1)
            if stack == 0:
                num_filters_out = num_filters_in * 4
                if block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if block == 0:
                    stride = (2, 2)

            block_result = conv_layer(inputs=data,
                                      nbr_filt=num_filters_in,
                                      kernel_size=1,
                                      activ=activation,
                                      strides=stride,
                                      batch_norm=batch_normalization,
                                      conv=False)
            block_result = conv_layer(inputs=block_result,
                                      nbr_filt=num_filters_in,
                                      kernel_size=3,
                                      conv=False)
            block_result = conv_layer(inputs=block_result,
                                      nbr_filt=num_filters_out,
                                      kernel_size=1,
                                      conv=False)

            if block == 0:
                data = conv_layer(inputs=data,
                                  nbr_filt=num_filters_out,
                                  kernel_size=1,
                                  strides=stride,
                                  activ=None,
                                  batch_norm=False)

            data = Add()([block_result, data])

        num_filters_in = num_filters_out

    data = BatchNormalization()(data)
    data = Activation('relu')(data)
    data = AveragePooling2D(pool_size=8)(data)
    output = Flatten()(data)
    output = Dense(nbr_classes, activation='softmax')(output)

    my_model = Model(inputs=inputs, outputs=output)
    return my_model

n = [2, 6, 12]
for nbr_blocks in n:
    depth = nbr_blocks * 9 + 2

    model = resnetV2(shape=input_shape, depth=depth, nbr_classes=10)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
    #model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(lr=1e-3, decay=1e-3/epochs), metrics=['accuracy'])
    model.summary()

    model_name = 'Resnet'+str(depth)+'v2_'+str(epochs)+'epoch_LRschedule'
    save_dir = os.path.join(os.getcwd(), 'SavedModels/'+model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path + '.h5')
    plot_model(model, to_file=model_path+ '.png', show_shapes=True, show_layer_names=False)


    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    # callbacks = [checkpoint]
    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        history=model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        history=model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['acc'])
    ax[1].plot(history.history['val_acc'])

    ax[0].set_xlim(0, epochs - 1)
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].set_title('Model accuracy - Train')

    ax[1].set_title('Model accuracy - Test')
    ax[1].set_xlim(0, epochs - 1)
    ax[1].set_ylabel('accuracy')
    ax[1].set_xlabel('epoch')

    fig.tight_layout()
    plt.savefig(model_path+'_accuracy', format='png')
    plt.close()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])

    ax[0].set_xlim(0, epochs - 1)
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].set_title('Model loss - Train')

    ax[1].set_title('Model loss - Test')
    ax[1].set_xlim(0, epochs - 1)
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')

    fig.tight_layout()
    plt.savefig(model_path + 'loss', format='png')
    plt.close()
