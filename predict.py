import numpy as np
import cv2
import os
import requests
import tflite_runtime.interpreter as tflite

MODEL_PATH = "models/model.tflite"
MODEL_URL = "https://github.com/Karanaldo-07/ball-bearing-defect-detection/releases/download/v1/model.tflite"


def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("Downloading model...")

        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

        print("Model downloaded.")


class BearingDefectPredictor:
    def __init__(self):
        download_model()

        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img.astype(np.float32), axis=0)
        return img

    def predict(self, img_path):
        input_data = self.preprocess(img_path)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        confidence = float(output[0])
        predicted_class = "Defective" if confidence > 0.5 else "OK"

        return {
            "class": predicted_class,
            "confidence": confidence,
            "raw_score": confidence
        }