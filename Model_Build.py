import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')


train_dir = "archive/train/"
test_dir = "archive/test"
valid_dir = 'archive/valid'

minority_train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.2,
        rotation_range=10,
        brightness_range=(0, 10),
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest',
)

main_train_gen = ImageDataGenerator(rescale=1./255)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

minority_class_directory = 'archive/train/1'

minority_class_generator = minority_train_gen.flow_from_directory(
    directory=minority_class_directory,
    target_size=(320, 320),
    batch_size=20,
    class_mode='binary',
    shuffle=True,
)

main_generator = main_train_gen.flow_from_directory(
    directory=train_dir,
    target_size=(320, 320),
    batch_size=20,
    class_mode='binary',
    shuffle=True,
)

val_images = main_train_gen.flow_from_directory(
        directory=valid_dir,
        target_size=(320, 320),
        class_mode='binary',
        shuffle=True,
)
test_images = test_gen.flow_from_directory(
        directory=test_dir,
        target_size=(320, 320),
        class_mode='binary',
        batch_size=1,
        shuffle=False,
)

pretrained_model = tf.keras.applications.ResNet101(
    input_shape=(320, 320, 3),
    include_top=False,
    weights='imagenet',
    pooling='max'
)
pretrained_model.trainable = False


inputs = pretrained_model.input

layer1 = tf.keras.layers.Dense(1024, activation='relu')(pretrained_model.output)
layer2 = tf.keras.layers.BatchNormalization()(layer1)
layer3 = tf.keras.layers.Dropout(0.4)(layer2)

layer4 = tf.keras.layers.Dense(512, activation='relu')(layer3)
layer5 = tf.keras.layers.BatchNormalization()(layer4)
layer6 = tf.keras.layers.Dropout(0.3)(layer5)

layer7 = tf.keras.layers.Dense(256, activation='relu')(layer6)
layer8 = tf.keras.layers.BatchNormalization()(layer7)
layer9 = tf.keras.layers.Dropout(0.2)(layer8)

outputs = tf.keras.layers.Dense(525, activation='softmax')(layer9)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(outputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


def balanced_generator(main_gen, minority_gen):
    while True:
        main_images, main_labels = next(main_gen)
        minority_images, minority_labels = next(minority_gen)
        combined_images = np.concatenate((main_images, minority_images), axis=0)
        combined_labels = np.concatenate((main_labels, minority_labels), axis=0)
        yield combined_images, combined_labels


history = model.fit(
    balanced_generator(main_generator, minority_class_generator),
    validation_data=val_images,
    epochs=5,
    steps_per_epoch=400,
)

model.save('transfer_resnet1.keras')
