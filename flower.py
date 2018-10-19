from keras.models import Sequential, Model
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

batch_size = 4
epochs = 300

input = Input(shape = (150, 150, 3), batch_shape = (batch_size, 150, 150, 3), name = 'input')
x = Conv2D(32, (3, 3), activation='relu')(input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(1, activation='sigmoid', name = 'output')(x)
model = Model(inputs = input, outputs= x)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'binary_flower/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'binary_flower/val',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=1340 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=64 // batch_size)
model.save('./weight/flower.h5')  # always save your weights after training or during training

# convert
output_names = [node.op.name for node in model.outputs]

export_dir = './weight/'
sess = K.get_session()
frozen_graphdef = tf.graph_util.convert_variables_to_constants(
      sess, sess.graph_def, output_names)
tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [input], [x])
open(export_dir+"flower.tflite", "wb").write(tflite_model)