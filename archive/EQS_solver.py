import time
import urllib.request

import cv2
import latex2mathml.converter
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
from tensorflow.keras.models import load_model

import helpers
from yolo_helper import Yolov4

names = [
    "+",
    "-",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "=",
    "a",
    "b",
    "c",
    "d",
    "x",
    "y",
    "z",
]


def process_str(x, i):
    list_type = []
    for a in x:
        if a.isalpha():
            list_type.append(2)
        if a.isnumeric():
            list_type.append(1)
        if a in ["+", "-"]:
            list_type.append(0)
    string = ""
    for b in range(len(list_type) - 1):
        if list_type[b] == 1 and list_type[b + 1] == 2:
            string += x[b] + "*"
        elif list_type[b] == 2 and list_type[b + 1] == 2:
            string += x[b] + "*"
        elif list_type[b] == 2 and list_type[b + 1] == 1:
            string += x[b] + "**"
        else:
            string += x[b]
    string += x[-1]

    if i == 1:
        if string[0] == "+":
            string = "-" + string[1:]
        else:
            string = "+" + string[1:]
    return string


def eq_solver(equation):
    list_eq = []
    for eq in equation:
        string = ""
        # left : 0 , right : 1
        for i, side in enumerate(eq.split("=")):
            if side[0] not in ["+", "-"]:
                side = "+" + side
            pos_sign = [i for i, char in enumerate(side) if char in ["+", "-"]]
            pos_sign.append(len(side))

            for j in range(len(pos_sign) - 1):
                string += process_str(side[pos_sign[j] : pos_sign[j + 1]], i)
        list_eq.append(string)
    print(list_eq)
    result = sympy.solve([parse_expr(i) for i in list_eq])

    # print(result)
    str_result = ""
    if type(result) == dict:
        for i, key in enumerate(result):
            if i == 0:
                if len(result) == 1:
                    str_result += "( {} = {} )".format(sympy.latex(key), sympy.latex(result[key]))
                else:
                    str_result += "( {} = {} ,".format(sympy.latex(key), sympy.latex(result[key]))
            elif i == len(result) - 1:
                str_result += " {} = {} )".format(sympy.latex(key), sympy.latex(result[key]))
            else:
                str_result += " {} = {} ,".format(sympy.latex(key), sympy.latex(result[key]))
        return [str_result]

    else:
        if len(result) == 0:
            return ["PHƯƠNG TRÌNH VÔ NGHIỆM"]
        list_result = []
        for r in result:
            str_result = ""
            for i, key in enumerate(r):
                if i == 0:
                    if len(r) == 1:
                        str_result += "( {} = {} )".format(sympy.latex(key), sympy.latex(r[key]))
                    else:
                        str_result += "( {} = {} ,".format(sympy.latex(key), sympy.latex(r[key]))
                elif i == len(r) - 1:
                    str_result += " {} = {} )".format(sympy.latex(key), sympy.latex(r[key]))
                else:
                    str_result += " {} = {} ,".format(sympy.latex(key), sympy.latex(r[key]))
            # Replace "//" to "/"
            list_result.append(str_result)
        return list_result


class Solver:
    def __init__(self, config_path, weight_eq, weight_char, weight_cnn):
        self.config_path = config_path
        self.weight_char = weight_char
        self.weight_eq = weight_eq
        self.weight_cnn = weight_cnn
        print("[INFO] Initializing...")
        t = time.time()
        self.model_cnn = load_model(self.weight_cnn)
        self.model_yolo_eq = Yolov4(self.weight_eq, self.config_path, "eq")
        self.model_yolo_char = Yolov4(self.weight_char, self.config_path, "char")
        print(f"[INFO] Done initialize in {time.time() - t}s")

    def soe_solver(self, input_image, label="img"):
        # đọc ảnh
        if label == "img":
            img_test = input_image
        elif label == "path":
            img_test = cv2.imread(input_image)
        else:
            raise "Invalid input type!!!"
        # 1  detect pt trong ảnh

        equation_coor = self.model_yolo_eq.detector(
            img_test, 0.5, 0.4
        )  # get coordinate of all eqs
        print(equation_coor)

        # 2 rotate img base on above coordinates
        image_skew = helpers.text_skew(img_test, equation_coor, False)

        # 3 Detect phương trình again
        equation_coor_1 = self.model_yolo_eq.detector(image_skew, 0.5, 0.4)  # 1-> new
        print(equation_coor_1)

        # Sắp xếp theo chiều từ trên xuống theo truc y
        equation_coor_1 = sorted(equation_coor_1, key=lambda x: x[1])

        # List equation images
        equation_image = [
            image_skew[y : (y + h), x : (x + w + 5)] for x, y, w, h in equation_coor_1
        ]
        # helpers.show_image(equation_image)

        list_text_equation = []
        #
        for num, eq in enumerate(equation_image):
            print(eq.shape)
            # 5  detect characters in each equations images
            char_coor = self.model_yolo_char.detector(eq, 0.5, 0.3)

            # List character images
            char_image = [eq[y : y + h, x : x + w] for x, y, w, h in char_coor]

            text = self.ocr(char_image)  # 6 classify character images
            list_text_equation.append(text)
            print(text)
            print("...................................")
            pass

        #
        print(list_text_equation)
        result = eq_solver(list_text_equation)  # 7 solve those equations
        print(result)

        # crop again for displaying on web
        eqs_cropped = helpers.text_skew(image_skew, equation_coor_1, True)

        return (
            eqs_cropped,
            [latex2mathml.converter.convert(r) for r in result],
            helpers.eq_4_display(list_text_equation),
        )

    def ocr(self, list_image):
        eq_str = ""
        for image in list_image:
            preprocess_image = helpers.preprocessing_image(image)
            y_predict = self.model_cnn.predict(preprocess_image.reshape(1, 28, 28, 1))
            text_predict = names[np.argmax(y_predict)]
            print(text_predict, np.max(y_predict))
            if np.max(y_predict) >= 0.5:
                eq_str += text_predict
        return eq_str


# if __name__ == "__main__":
#     print(eq_solver(['2a2+4b=6', 'a-b+5c=5', 'b=2c-1']))
