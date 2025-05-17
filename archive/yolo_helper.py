import time

import cv2
import numpy as np

# print(cv2.__version__)
# print(matplotlib.__version__)

# %cd / content / drive / My Drive / Final Project / Khang


class Yolov4:
    def __init__(self, weight_path, config_path, label):
        self.weight_path = weight_path
        self.config_path = config_path
        self.label = label
        self.net = self.creat_net()

    def detector(self, image, confidence_threshold, NMS_threshold):
        self.image = image
        h, w, _ = self.image.shape
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(self.image, 1 / 255., (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)
        layer_outputs = self.net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:

                    center_x, center_y, width, height = list(map(int, detection[0:4] * [w, h, w, h]))

                    top_left_x = int(center_x - (width / 2))
                    top_left_y = int(center_y - (height / 2))

                    boxes.append([top_left_x, top_left_y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, NMS_threshold)
        self.num_obj = len(indices)

        list_coor = []
        crop_scale = 0.05
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                x = abs(int(x - crop_scale * w))
                y = abs(int(y - crop_scale * h))
                w = abs(int((1 + 2 * crop_scale) * w))
                h = abs(int((1 + 2 * crop_scale) * h))

                list_coor.append((x, y, w, h))

        list_coor = sorted(list_coor, key=lambda x: x[0])

        return [[x, y, w, h] for x, y, w, h in list_coor]

    def creat_net(self):
        print("[INFO] Start reading net...")
        t = time.time()
        net = cv2.dnn.readNetFromDarknet(self.config_path, self.weight_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"[INFO] Done reading net!\n Reading time: {time.time() - t}s")
        return net

    def __str__(self):
        return "Number of object detected: {}".format(self.num_obj)

    def num(self):
        return self.num_obj
