import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

path = 'classification'
print(os.listdir(path))

for i in os.listdir(os.path.join(path,'seg_train/seg_train')):
    print(i)
batch_size =20
img_height = 256
img_width = 256
IMAGE_SIZE =(256,256)


def load_data():


    datasets = ['classification']
    output = []

    # Iterate through training and test sets
    for dataset in datasets:

        images = []
        labels = []

        print("Loading {}".format(dataset))

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output

train_data_dir = r"C:\Users\atuly\PycharmProjects\Yolov7\classification\seg_train\seg_train"
val_data_dir = r"C:\Users\atuly\PycharmProjects\Yolov7\classification\seg_test\seg_test"

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_data_dir,
  validation_split=0.01,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# validation data batches
val_ds = tf.keras.utils.image_dataset_from_directory(
  val_data_dir,
  validation_split=0.99,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


print('Number of train batches: %d' % tf.data.experimental.cardinality(train_ds))

print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds))
Data_augment = keras.Sequential([
    tf.keras.layers.Rescaling(1/255.0),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomFlip(),
]
)

def convolutional_model():
    # START CODE HERE
    model = tf.keras.models.Sequential([
        Data_augment,
        tf.keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(256,256, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (5, 5), padding='valid', activation='relu'),
        tf.keras.layers.MaxPooling2D(3, 3),
        tf.keras.layers.Conv2D(128, (1, 1), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, (1, 1), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='valid', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, (1, 1), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(128, (1, 1), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(512, (1, 1), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

model=convolutional_model()


model.fit(train_ds,validation_data=val_ds,epochs=20)
