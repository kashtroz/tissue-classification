import tensorflow as tf
import numpy as np

class Tester:
    def __init__(self, model_path):
        self.model_path = model_path

    def test(self, image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(180, 180))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        model = tf.keras.models.load_model(self.model_path)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)

        predicted_labels = np.argmax(predictions, axis=1)

        return predictions, predicted_labels
