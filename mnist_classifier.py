import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 

#Cargar el dataset MNIST

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test , y_test) = mnist.load_data()

# Normalizamos los datos

x_train, x_test = x_train / 255.0, x_test / 255.0

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"{y_train[i]}")
    plt.axis('off')
plt.show()

# Definimos el modelo

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilamos el modelo

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamos el modelo

model.fit(x_train, y_train, epochs=5)

# Evaluamos el modelo

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n Precision en el conjunto de prueba: {test_acc:.4f}")

# Guardamos el modelo para un uso posterior

model.save('mnist_model.h5')

