import os
import time
import urllib.request

import cv2
import latex2mathml.converter
import matplotlib

import EQS_solver as eq

print(cv2.__version__)
print(matplotlib.__version__)


equation_weight = r"weights/yolov4_training_2000_eq.weights"
char_weight = r"weights/yolov4_training_2000_char.weights"
config_path = r"darknet/cfg/yolov4_training.cfg"
ocr_path = r"weights/model_ocr.h5"

# testing

solver = eq.Solver(config_path, equation_weight, char_weight, ocr_path)
file_name = "123.jpg"
path = "static/uploads"
img_path = os.path.join(path, file_name)
test_img = cv2.imread(img_path)
output = solver.soe_solver(test_img)
cv2.imshow("test", output[0])
print(output[1:])
cv2.waitKey()
cv2.destroyAllWindows()
