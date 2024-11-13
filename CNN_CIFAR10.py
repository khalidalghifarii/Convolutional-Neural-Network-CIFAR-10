import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize image data
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to categorical
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Build CNN model
model = tf.keras.Sequential([
    # First convolution layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Third convolution layer
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten layer
    tf.keras.layers.Flatten(),

    # Dense layer with Dropout
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(10, activation='softmax')
])

# Print model summary
model.summary()

# Compile model
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# Train model
history = model.fit(train_images, train_labels, 
                   epochs=10, 
                   batch_size=64, 
                   validation_data=(test_images, test_labels))

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Save model
model.save('cifar10_cnn_model.h5')

# Function for image prediction
def predict_image(file_path):
    # Load and prepare image
    img = Image.open(file_path)
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name, prediction[0][predicted_class_index]