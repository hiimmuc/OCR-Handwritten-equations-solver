import os
import time
import urllib.request

import cv2
import latex2mathml.converter
import matplotlib.pyplot as plt
import numpy as np


def preprocessing_image(image):
    """
    preprocess character's images
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.medianBlur(image_gray, 3)
    ret, thresh = cv2.threshold(image_blur, 120, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_bw = cv2.bitwise_not(thresh)

    # Padding
    h, w = image_gray.shape
    pad_size = int(abs(h - w) / 2)
    pad_extra = abs(h - w) % 2
    image_padding = np.array([])
    if h > w:
        image_padding = cv2.copyMakeBorder(image_bw,
                                           0,
                                           0,
                                           pad_size + pad_extra,
                                           pad_size,
                                           cv2.BORDER_CONSTANT,
                                           value=[0, 0, 0])
        image_padding = cv2.copyMakeBorder(image_padding,
                                           5,
                                           5,
                                           5,
                                           5,
                                           cv2.BORDER_CONSTANT,
                                           value=[0, 0, 0])
    elif w > h:
        image_padding = cv2.copyMakeBorder(image_bw,
                                           pad_size + pad_extra,
                                           pad_size,
                                           0,
                                           0,
                                           cv2.BORDER_CONSTANT,
                                           value=[0, 0, 0])
        image_padding = cv2.copyMakeBorder(image_padding,
                                           5,
                                           5,
                                           5,
                                           5,
                                           cv2.BORDER_CONSTANT,
                                           value=[0, 0, 0])
    else:
        image_padding = image_bw

    # Resize 28x28
    image_resized = cv2.resize(image_padding, (28, 28))

    # Dilate
    kernel_dilate = np.array((3, 3), np.uint8)
    image_dilate = cv2.dilate(image_resized, kernel_dilate)

    # show_image([image_dilate])
    # plt.show()

    # Expand dims
    image_result = np.expand_dims(image_dilate, 0)
    return image_result


def text_skew(image, coor, check):
    print(coor)

    np_coor = np.array(coor)
    x_new = np.min(np_coor[:, 0])
    y_new = np.min(np_coor[:, 1])
    w_new = np.max(np.array([x + w for x, y, w, h in coor])) - x_new
    h_new = np.max(np_coor[:, 1]) + np_coor[np.argmax(
        np_coor[:, 1])][3] - np.min(np_coor[:, 1])

    # Crop soe
    soe_cropped = image[y_new:y_new + h_new, x_new:x_new + w_new]
    # If check == true thì chỉ trả về ảnh đc crop
    if check == True:
        return soe_cropped
    # if check == False thì sẽ trả về ảnh gốc đã được xoay
    else:
        gray = cv2.cvtColor(soe_cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        image_black = np.zeros((image.shape[0], image.shape[1]))
        image_black[y_new:y_new + h_new, x_new:x_new + w_new] = thresh

        coords = np.column_stack(np.where(image_black > 0))
        angle = cv2.minAreaRect(coords)[-1]
        # the `cv2.minAreaRect` function returns values in the
        # range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we
        # need to add 90 degrees to the angle
        if angle < -45:
            angle = -(90 + angle)
        # otherwise, just take the inverse of the angle to make
        # it positive
        else:
            angle = -angle
        print(angle)
        # if abs(angle) < 10 :
        #   return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image,
                                 M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

        return rotated


def eq_4_display(list_eq):
    list_4_display = []
    for eq in list_eq:
        string = ""
        for i, char in enumerate(eq):
            if char in ["+", "="]:
                string += " " + char + " "
            elif char == "-":
                if i == 0 or string[len(string) - 2:len(string)] == "= ":
                    string += char
                else:
                    string += " " + char + " "
            elif char.isnumeric():
                if i == 0:
                    string += char
                elif string[-1].isalpha():
                    string += "^" + char
                else:
                    string += char
            else:
                string += char
            # print(string)
        mathml_output = latex2mathml.converter.convert(string)
        list_4_display.append(mathml_output)
    return list_4_display


def show_image(image_list):
    for image in image_list:
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.axis('off')
