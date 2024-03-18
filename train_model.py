import keras

# Prepare data
dataset_mnist = keras.datasets.mnist
(training_images, training_labels), (testing_images, testing_labels) = dataset_mnist.load_data()
training_images = keras.utils.normalize(training_images, axis=1)
testing_images = keras.utils.normalize(testing_images, axis=1)

# Neural network
model = keras.models.Sequential()
model.add(keras.layers.Flatten())  # 28x28 -> 1x784
model.add(keras.layers.Dense(128, activation=keras.activations.relu))
model.add(keras.layers.Dense(128, activation=keras.activations.relu))
model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

# Train model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=3)

# Print results
val_loss, val_acc = model.evaluate(testing_images, testing_labels)
print("Feilmargin", val_loss * 100, "%")
print("Nøyaktighet", val_acc * 100, "%")

# Save model
model.save_weights('digit_model.weights.h5')
with open('digit_model.architecture.json', 'w') as f:
    f.write(model.to_json())
