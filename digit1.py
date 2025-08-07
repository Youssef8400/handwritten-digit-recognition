import tensorflow as tf
from keras._tf_keras.keras.models import Model, load_model
from keras._tf_keras.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

print(f"TensorFlow version: {tf.__version__}")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

def residual_block(x, filters, downsample=False):
    identity = x
    stride = 2 if downsample else 1

    x = Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if downsample or identity.shape[-1] != filters:
        identity = Conv2D(filters, 1, strides=stride, padding='same')(identity)
        identity = BatchNormalization()(identity)

    x = Add()([x, identity])
    x = ReLU()(x)
    return x

def build_resnet10(input_shape=(28, 28, 1), num_classes=10):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='ResNet10_MNIST')
    return model

model = build_resnet10()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train_cat, epochs=5, validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Test accuracy: {test_acc*100:.2f}%")

model.save("cnn_m.h5")
print("✅ Modèle sauvegardé sous 'cnn_m.h5'.")
