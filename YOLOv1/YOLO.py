import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras.regularizers import l2
from tensorflow.math import exp as expo
from keras.preprocessing.image import ImageDataGenerator



def lr_schedule(epoch):
    """Learning Rate Schedule
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    if epoch > 135:
        lr = 1e-4
    elif epoch <= 135 and epoch > 105:
        lr = 1e-3
    elif epoch <= 105 and epoch > 75:
        lr = 1e-2
    elif epoch <= 75:
        lr = 1e-3+ 0.009 * expo(0.1*(epoch - 75))

    print('Learning rate: ', lr)
    return lr

def conv_block(inputs, num_filters, kernel=1, strides=(1,1), padding= 'same', name=''):
    if strides==(2,2):
        padding = 'valid'
    layer = Conv2D(filters= num_filters, kernel_size=kernel, strides=strides, padding=padding, kernel_regularizer=l2(0.005), name=name)(inputs)
    layer = BatchNormalization()(layer)
    layer= LeakyReLU(alpha=0.1)(layer)

    return layer

#Stride = 2 divides the output dimension by 2
#Input size = 448x448x(3?)
def YOLO(inputs, S, B, C):
    result = Conv2D(filters= 64, kernel_size=S ,strides=(2, 2))(inputs)
    result = MaxPooling2D(pool_size=(2,2), strides=(2,2))(result)

    result = Conv2D(filters=192, kernel_size=3)(result)
    result = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(result)

    result = Conv2D(filters=128, kernel_size=1)(result)
    result = Conv2D(filters=256, kernel_size=3)(result)
    result = Conv2D(filters=256, kernel_size=1)(result)
    result = Conv2D(filters=512, kernel_size=3)(result)
    result = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(result)

    result = Conv2D(filters=256, kernel_size=1)(result)
    result = Conv2D(filters=512, kernel_size=3)(result)
    result = Conv2D(filters=256, kernel_size=1)(result)
    result = Conv2D(filters=512, kernel_size=3)(result)
    result = Conv2D(filters=256, kernel_size=1)(result)
    result = Conv2D(filters=512, kernel_size=3)(result)
    result = Conv2D(filters=256, kernel_size=1)(result)
    result = Conv2D(filters=512, kernel_size=3)(result)
    result = Conv2D(filters=512, kernel_size=1)(result)
    result = Conv2D(filters=1024, kernel_size=3)(result)
    result = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(result)

    result = Conv2D(filters=512, kernel_size=1)(result)
    result = Conv2D(filters=1024, kernel_size=3)(result)
    result = Conv2D(filters=512, kernel_size=1)(result)
    result = Conv2D(filters=1024, kernel_size=3)(result)
    result = Conv2D(filters=1024, kernel_size=3)(result)
    result = Conv2D(filters=1024, kernel_size=3, strides=(2,2))(result)

    result = Conv2D(filters=1024, kernel_size=3)(result)
    result = Conv2D(filters=1024, kernel_size=3)(result)

    result = Flatten()(result)
    result = Dense(units=4096)(result)
    result = Dropout(0.5)(result)
    result = Dense(units=S*S*(5*B + C))(result)

    my_model = Model(inputs=inputs, outputs=result)
    return my_model

"""Output is S x S x (5B + C)
S: We divide image in a SxS grid
B: Number ob bounding boxes predicted for every grid cell
C: Number of classes

For PASCAL VOC, S = 7, B = 2 and C = 20 ==> 7 x 7 x 30

"""

epochs = 160
batch_size = 64
momentum = 0.9
decay = 0.0005
data_augmentation = True

































if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
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
        width_shift_range=0.2,
        # randomly shift images vertically
        height_shift_range=0.2,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=1.5,#0
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
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)