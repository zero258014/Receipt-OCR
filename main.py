from flask import Flask, request
from flask import render_template
import settings
import utils
import numpy as np
import cv2
import receipt_predictions as pred
import receipt_imgPreProcess as prep


app = Flask(__name__)
app.secret_key = "receipt_sacnner_app"

receipt_scan = utils.ReceiptScan()


@app.route("/", methods=["GET", "POST"])
def scanimg():
    if request.method == "POST":
        file = request.files["image_name"]
        upload_image_path = utils.save_upload_image(file)
        print("saved in :", upload_image_path)

        # レシートの4点の座標とサイズの取得
        four_points, size = receipt_scan.document_scanner(upload_image_path)
        print(four_points, size)
        if four_points is None:
            message = "レシートの座標を見つからないので、ランダムの座標を表示する"
            points = [
                {"x": 10, "y": 10},
                {"x": 120, "y": 10},
                {"x": 120, "y": 120},
                {"x": 10, "y": 120},
            ]

            return render_template("scanner.html",
                                   points=points,
                                   fileupload=True,
                                   message=message)

        else:
            points = utils.array_to_json_format(four_points)
            message = "opencvを使って、レシートの座標を見つかった"

            return render_template("scanner.html",
                                   points=points,
                                   fileupload=True,
                                   message=message)

        return render_template("scanner.html")
    return render_template("scanner.html")


@app.route("/transform", methods=["POST"])
def transform():
    try:
        points = request.json["data"]
        array = np.array(points)
        magic_color = receipt_scan.calibrate_to_original_size(array)
        # utils.save_upload_image(magic_color, "magic_color.jpg")
        filename = "magic_color.jpg"
        magic_image_path = settings.join_path(settings.MEDIA_DIR, filename)
        cv2.imwrite(magic_image_path, magic_color)

        return "sucess"
    except:
        return "fail"


@app.route("/prediction", )
def prediction():
    # 識別する
    wrap_image_path = settings.join_path(settings.MEDIA_DIR, "magic_color.jpg")
    img = cv2.imread(wrap_image_path)

    img_prep = prep.process(img)
    img_PIL = prep.cvToPil(img_prep)
    img_dpi300_path = settings.join_path(settings.MEDIA_DIR, "dpi300.jpg")
    img_PIL.save(img_dpi300_path, dpi=(300, 300))

    img_dpi300 = cv2.imread(img_dpi300_path)

    img_bb, entities, result_text = pred.getPredictions(img_dpi300)

    bb_filename = settings.join_path(settings.MEDIA_DIR, "bounding_box.jpg")
    cv2.imwrite(bb_filename, img_bb)
    return render_template("result.html", results=entities, result_text=result_text)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, host='localhost')
