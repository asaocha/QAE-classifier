import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import utils
import numpy as np

class FCNN:
    def __init__(self):
        self.model = None

    def train(self, 
              train_images, train_labels, 
              training_epochs,
              label_num, k):
        
        input_size = int(np.sqrt(train_images.shape[1]))
        
        self.model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size, input_size)),
        tf.keras.layers.Dense(k, activation='relu'),
        tf.keras.layers.Dense(label_num, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

        train_images = train_images.reshape(-1, input_size, input_size).copy()
        train_labels_categorical = utils.to_categorical(train_labels)

        self.model.fit(train_images, train_labels_categorical, epochs=training_epochs, validation_split=0.33)

    
    def test(self, test_images, test_labels):

        input_size = int(np.sqrt(test_images.shape[1]))
        
        test_images = test_images.reshape(-1, input_size, input_size).copy()
        test_labels_categorical = utils.to_categorical(test_labels)

        _, test_accuracy = self.model.evaluate(test_images, test_labels_categorical)
        
        return test_accuracy
