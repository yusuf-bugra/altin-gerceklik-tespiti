import os
os.environ["GDOWN_CACHE_DIR"] = "/tmp/gdown"  # MUTLAKA EN ÜSTE

from flask import Flask, request, render_template
import gdown
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = "https://drive.google.com/uc?id=1beF5ywhTyYtLyc_aL5Hfs0dUFQZc2H1Y"
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_training5.pth")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CONFIG_PATH = "detectron2_config.yaml"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, use_cookies=False)

cfg = get_cfg()
cfg.merge_from_file(CONFIG_PATH)
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)
MetadataCatalog.get("my_dataset").thing_classes = ["fake_gold", "real_gold"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    if request.method == "POST":
        f = request.files["file"]
        filename = secure_filename(f.filename)
        filepath = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        f.save(filepath)

        im = cv2.imread(filepath)
        outputs = predictor(im)
        pred_class = outputs["instances"].pred_classes[0].item()
        
        scores = outputs["instances"].scores
        confidence = scores[0].item() * 100
        
        result = "Gerçek Altın" if pred_class == 1 else "Sahte Altın"
    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
