from base64 import b64encode
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from torchvision import transforms
from werkzeug.exceptions import abort
from wtforms import FileField, SubmitField

from flask import render_template, Response, flash, jsonify
from app.main import main_blueprits
from app.source.utils import classify_image


# from source.test_new_images import detect_mask_in_image


@main_blueprits.route("/")
def home_page():
    return render_template("home_page.html")


def allowed_file(file):
    extention_file = file.split(".")[-1]
    return extention_file in ['jpg', 'jpeg', 'png']


@main_blueprits.route("/image-detector", methods=["GET", "POST"])
def image_mask_detection():
    return render_template("image_detector.html",
                           form=PhotoMaskForm())


@main_blueprits.route("/image_processing_old", methods=["POST"])
def image_processing_old():
    form = PhotoMaskForm()

    if not form.validate_on_submit():
        flash("An error occurred", "danger")
        abort(Response("Error", 400))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    pil_image = Image.open(form.image.data)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    label, _ = classify_image(pil_image.convert('RGB'), transform)

    image_detected = Image.fromarray(image, 'RGB')
    # image_detected = Image.fromarray(rgb_image, 'RGB')

    with BytesIO() as img_io:
        image_detected.save(img_io, 'PNG')
        img_io.seek(0)
        base64img = "data:image/png;base64," + b64encode(img_io.getvalue()).decode('ascii')
        return base64img, label


@main_blueprits.route("/image_processing", methods=["POST"])
def image_processing():
    form = PhotoMaskForm()

    if not form.validate_on_submit():
        flash("An error occurred", "danger")
        abort(Response("Error", 400))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    pil_image = Image.open(form.image.data)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    label, _ = classify_image(pil_image.convert('RGB'), transform)

    image_detected = Image.fromarray(image, 'RGB')

    with BytesIO() as img_io:
        image_detected.save(img_io, 'PNG')
        img_io.seek(0)
        base64img = "data:image/png;base64," + b64encode(img_io.getvalue()).decode('ascii')

    return jsonify({"image": base64img, "label": label})


class PhotoMaskForm(FlaskForm):
    image = FileField('Choose image:',
                      validators=[
                          FileAllowed(['jpg', 'jpeg', 'png'], 'The allowed extensions are: .jpg, .jpeg and .png')])

    submit = SubmitField('Classification')
    cancel = SubmitField('Cancel')
