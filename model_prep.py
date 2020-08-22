import os
import tensorflow as tf
import tensorflow_hub as hub 

model = hub.load("https://tfhub.dev/google/elmo/3")

sign = {
    "serving_default": model.signatures["default"]
}

export = os.path.join(os.getcwd(), "model/1")

tf.saved_model.save(model, export, sign)