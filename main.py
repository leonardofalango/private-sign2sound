import logging

logging.basicConfig(
    filename='var/logs.log', level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


import tensorflow as tf
from tensorflow.keras import layers, models

# Carregar o conjunto de dados
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalizar os valores dos pixels para o intervalo [0,1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Definir o modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Treinar o modelo
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Avaliar a precisão do modelo
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


