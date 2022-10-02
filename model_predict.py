import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X):
         
        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)

        return out