import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

path = r'test1.png'
test_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (28, 28))
test_image = test_image.astype('float32')
test_image = test_image.reshape(1, 28, 28, 1)
test_image = 255-test_image
test_image /= 255


plt.imshow(test_image[0], cmap=plt.cm.binary)
plt.show()

model = tf.keras.models.load_model('tallgjennkjenner.model')
predictions = model.predict(test_image)

print(np.argmax(predictions[0]))
