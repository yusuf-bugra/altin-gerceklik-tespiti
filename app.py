from flask import Flask, render_template, request
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os

app = Flask(__name__)

cfg = get_cfg()
cfg.merge_from_file("detectron2_config.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.WEIGHTS = "model_training5.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
predictor = DefaultPredictor(cfg)
MetadataCatalog.get("my5_test").set(thing_classes=["fake", "real"])

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        file_path = os.path.join("static", "upload.jpg")
        file.save(file_path)

        im = cv2.imread(file_path)
        outputs = predictor(im)
        instances = outputs["instances"]
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        if scores:
            en_yuksek = max(zip(scores, classes), key=lambda x: x[0])
            result = {
                "sonuc": "Gerçek Altın" if en_yuksek[1] == 1 else "Sahte Altın",
                "guven": f"{en_yuksek[0]*100:.2f}%"
            }
        else:
            result = {"sonuc": "Hiçbir şey tespit edilemedi", "guven": "0%"}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
