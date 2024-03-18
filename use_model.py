import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import model_from_json


def get_image(target_path):
    color_image = cv2.imread(target_path, cv2.IMREAD_ANYCOLOR)
    color_image = cv2.resize(color_image, (28, 28))
    grey_image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    grey_image = cv2.resize(grey_image, (28, 28))
    grey_image = grey_image.astype('float32')
    image = grey_image.reshape(1, 28, 28, 1)
    image = 255 - image
    image /= 255
    return image, grey_image, color_image


# Load data
file_path = r'test1.png'
test_data = get_image(file_path)
test_image = test_data[0]

# Load model
with open('digit_model.architecture.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('digit_model.weights.h5')

# Use model
predictions = model.predict([test_image])
test_result = np.argmax(predictions[0])
print("Resultat", test_result)

# Show results
plt.subplot(2, 1, 1)
plt.xticks([])
plt.yticks([])
plt.imshow(test_data[2], cmap=plt.cm.binary)
plt.xlabel("Test bilde")
plt.subplot(2, 1, 2)
plt.xticks([])
plt.yticks([])
plt.imshow(test_data[1], cmap=plt.cm.binary)
plt.xlabel(f"Resultat: {test_result}")
plt.show()
