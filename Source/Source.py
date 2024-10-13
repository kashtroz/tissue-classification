import tensorflow as tf
import matplotlib.pyplot as plt
from Splitter import Splitter
from ImageProcessing import ImgProcessingandTraining
from Displayer import Displayer
from Tester import Tester
import numpy as np
import warnings
warnings.filterwarnings('ignore')

path = r"DataStore/EndoscopicBladderTissue"
img_height = 180
img_width = 180
batch_size = 32
epochs = 100


#Data Loading and Splitting
splitter = Splitter(path)
train_ds = splitter.train_data(batch_size=batch_size, img_width=img_width, img_height=img_height)
val_ds = splitter.val_data(batch_size=batch_size, img_width=img_width, img_height=img_height)

class_names = train_ds.class_names
print(class_names)

for image_batch, label_batch in train_ds:
    print(image_batch.shape)
    print(label_batch.shape)
    break

preprocessor_trainer = ImgProcessingandTraining(train_ds, val_ds)
train_data, val_data, num_classes = preprocessor_trainer.preprocessing(class_names=class_names)
model = preprocessor_trainer.build_model(num_classes, img_height=img_height, img_width=img_width)

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()



model_path = r"C:\Users\Devadath Ram\PycharmProjects\pythonProject4\Source\Models\LatestModel.keras"
model = tf.keras.models.load_model(r"C:\Users\Devadath Ram\PycharmProjects\pythonProject4\Source\Models\LatestModel.keras")
test_data_path = r"C:\Users\Devadath Ram\PycharmProjects\pythonProject4\Source\DataStore\TestDirectory\NST\cys_case_0_pt02_0115.png"
tester = Tester(model_path)
prediction, prediction_labels = tester.test(test_data_path)
logits = np.array(prediction)


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


probabilities = softmax(logits)
predicted_label = np.argmax(probabilities)

print("Probabilities:", probabilities)
print("Predicted Label Index:", predicted_label)

displayer = Displayer(class_names)
displayer.test_display(test_data_path, predicted_label)

"""
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print("Accuracy of the model: ", acc)
print("Validation accuracy: ", val_acc)

model.save(r"C:/Users\Devadath Ram\PycharmProjects\pythonProject4\Source\Models\LatestModel.keras")
"""



