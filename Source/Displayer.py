import matplotlib.pyplot as plt
import tensorflow as tf

class Displayer:
    def __init__(self, class_names):
        self.class_names = class_names

    def test_display(self, test_image_path, predicted_label_index):
        img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(180, 180))
        predicted_label = self.class_names[predicted_label_index]
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis("off")
        plt.show()

