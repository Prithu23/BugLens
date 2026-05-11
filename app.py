import io
import base64
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
model = YOLO("model1.pt")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")

    results = model(img)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = results.names[cls]
        detections.append({"label": name, "confidence": round(conf * 100, 1)})

    annotated = Image.fromarray(results.plot())
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({"detections": detections, "image": img_b64})

if __name__ == "__main__":
    app.run(debug=True)
