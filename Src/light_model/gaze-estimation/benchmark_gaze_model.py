import os
import cv2
import time
import numpy as np
import onnxruntime as ort

DATASET_DIR = "data"
# MODEL_PATH = "weights/mobilenetv2_gaze.onnx"
MODEL_PATH = "weights/mobileone_s0_gaze.onnx"

INPUT_SIZE = (448, 448)

LABELS = ["left", "right", "center", "up", "down"]


class GazeBenchmark:

    def __init__(self, model_path):

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.model = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        self.thresholds = {
            "yaw": 20,
            "pitch": 20
        }

    def preprocess(self, img):

        img = cv2.resize(img, INPUT_SIZE)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))

        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, img):

        input_tensor = self.preprocess(img)

        input_name = self.model.get_inputs()[0].name

        start = time.time()

        outputs = self.model.run(None, {input_name: input_tensor})

        inference_time = time.time() - start

        yaw_pred = outputs[0][0]
        pitch_pred = outputs[1][0]

        yaw_exp = np.exp(yaw_pred - np.max(yaw_pred))
        yaw_prob = yaw_exp / np.sum(yaw_exp)

        pitch_exp = np.exp(pitch_pred - np.max(pitch_pred))
        pitch_prob = pitch_exp / np.sum(pitch_exp)

        bins = np.arange(len(yaw_prob))
        yaw = np.sum(yaw_prob * bins) * 4 - 180

        bins = np.arange(len(pitch_prob))
        pitch = np.sum(pitch_prob * bins) * 4 - 180

        print(f"yaw:{yaw:.2f} pitch:{pitch:.2f}")

        return pitch, yaw, inference_time

    def get_direction(self, pitch, yaw):
        if abs(yaw) > abs(pitch):

            if abs(yaw) > self.thresholds["yaw"]:
                return "right" if yaw > 0 else "left"

        else:

            if abs(pitch) > self.thresholds["pitch"]:
                return "down" if pitch > 0 else "up"

            return "center"

def benchmark():

    model = GazeBenchmark(MODEL_PATH)

    total = 0
    correct = 0
    total_time = 0

    for label in LABELS:

        folder = os.path.join(DATASET_DIR, label)

        files = os.listdir(folder)

        for file in files:

            path = os.path.join(folder, file)

            img = cv2.imread(path)

            if img is None:
                continue

            pitch, yaw, t = model.predict(img)

            pred = model.get_direction(pitch, yaw)

            total += 1
            total_time += t

            if pred == label:
                correct += 1

            print(f"{total} | GT:{label} | Pred:{pred}")

    avg_time = total_time / total
    fps = 1 / avg_time
    accuracy = correct / total

    print("\n======= BENCHMARK RESULT =======")

    print("Total images:", total)

    print(f"Average inference time: {avg_time*1000:.2f} ms")

    print(f"FPS: {fps:.2f}")

    print(f"Direction Accuracy: {accuracy*100:.2f}%")

    print("================================")


if __name__ == "__main__":

    benchmark()
