import os

import cv2

# import latex2mathml.converter
from flask import (
    Flask,
    Markup,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from sympy.parsing.sympy_parser import parse_expr
from werkzeug.utils import secure_filename

import EQS_solver as EQ_

equation_weight = r"weights/yolov4_training_2000_eq.weights"
char_weight = r"weights/yolov4_training_2000_char.weights"
config_path = r"backup/cfg/yolov4_training.cfg"
ocr_path = r"weights/model_ocr.h5"

# testing
solver = EQ_.Solver(config_path, equation_weight, char_weight, ocr_path)
# app
app = Flask(__name__, template_folder=r"templates")
UPLOAD_FOLDER = "static/uploads"
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/images/<filename>")
def images(filename):
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    return send_file(image_path)


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files["file"]
        print(file)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            full_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(full_path)
            print("123", full_path)
            # Predict
            soe_image, result, list_text_equation = solver.soe_solver(full_path, label="path")
            # Image cropped
            cv2.imwrite(app.config["UPLOAD_FOLDER"] + "/cropped_" + filename, soe_image)
            soe_image = cv2.copyMakeBorder(
                soe_image, 0, 0, 10, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            image_cropped_url = url_for("images", filename="cropped_" + filename)
            # Original image
            original_image = cv2.imread(full_path)
            original_image = cv2.copyMakeBorder(
                original_image, 0, 0, 0, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            cv2.imwrite(app.config["UPLOAD_FOLDER"] + "/original_" + filename, original_image)
            original_image_url = url_for("images", filename="original_" + filename)

            return render_template(
                "upload.html",
                original_image=original_image_url,
                cropped_image=image_cropped_url,
                result=result,
                text=list_text_equation,
            )
    else:
        return redirect("/")


if __name__ == "__main__":
    app.run()
