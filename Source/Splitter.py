import tensorflow as tf


class Splitter:
    def __init__(self, path):
        self.path = path

    def train_data(self, img_height, img_width, batch_size):
        train_ds = tf.keras.utils.image_dataset_from_directory(self.path,
                                                               validation_split=0.2,
                                                               subset='training',
                                                               seed=123,
                                                               image_size=(img_height, img_width),
                                                               batch_size=batch_size)
        return train_ds

    def val_data(self, img_height, img_width, batch_size):
        val_ds = tf.keras.utils.image_dataset_from_directory(self.path,
                                                             validation_split=0.2,
                                                             subset='validation',
                                                             seed=123,
                                                             image_size=(img_height, img_width),
                                                             batch_size=batch_size)
        return val_ds


