import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class ImgProcessingandTraining:
    def __init__(self, train_ds, val_ds):
        self.train_ds = train_ds
        self.val_ds = val_ds

    def preprocessing(self, class_names):
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        print(np.min(first_image), np.max(first_image))
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        num_classes = len(class_names)

        return train_ds, val_ds, num_classes

    def build_model(self, num_classes, img_height, img_width):
        model = Sequential([
          layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
          layers.Conv2D(16, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Flatten(),
          layers.Dense(128, activation='relu'),
          layers.Dense(num_classes)
        ])
        return model
