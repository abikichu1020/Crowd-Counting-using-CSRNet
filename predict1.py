import os
import torch
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from io import BytesIO
from model import CSRNet  # Import your CSRNet model

# Flask setup
app = Flask(__name__, static_folder="static")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CSRNet().to(device)
checkpoint = torch.load("PartAmodel_best.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Image preprocessing
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (1024, 768))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img_resized).unsqueeze(0), img_resized

# Prediction + Visualization
def predict_and_visualize(image_path):
    img_tensor, orig_img = load_image(image_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)

    density_map = output.squeeze().cpu().numpy()
    density_map = gaussian_filter(density_map, sigma=1)
    density_map = np.clip(density_map, 0, None)
    count = int(round(np.sum(density_map)))

    # Create density map image in memory
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.imshow(density_map, cmap='jet')
    axs.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="jpeg", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    density_map_b64 = base64.b64encode(buf.read()).decode()
    plt.close()

    # Create overlay image
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.imshow(orig_img)
    axs.imshow(density_map, cmap='jet', alpha=0.5)
    axs.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="jpeg", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    overlay_b64 = base64.b64encode(buf.read()).decode()
    plt.close()

    return count, density_map_b64, overlay_b64

# HTML Page
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Crowd Counting</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 900px; margin: auto; padding: 20px; background: #f8f8f8; }
  h1 { text-align: center; color: #333; }
  .box { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 0 5px #ccc; text-align: center; }
  input[type="file"], button { margin: 10px 0; width: 100%; padding: 10px; font-size: 15px; }
  button { background: #4a90e2; color: white; border: none; cursor: pointer; }
  img { max-width: 100%; margin-top: 10px; border-radius: 5px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 20px; }
</style>
</head>
<body>

<h1>Crowd Counting with CSRNet</h1>
<div class="box">
  <form id="uploadForm">
    <input type="file" id="imageInput" accept="image/*" required>
    <button type="submit">Detect</button>
  </form>

  <div id="loading" style="display:none;">Processing...</div>

  <div id="result" style="display:none;">
    <p>Estimated Count: <strong id="count">0</strong></p>
    <div class="grid">
      <div>
        <h3>Original</h3>
        <img id="originalImage" />
      </div>
      <div>
        <h3>Density Map</h3>
        <img id="densityMap" />
      </div>
      <div>
        <h3>Overlay</h3>
        <img id="overlayImage" />
      </div>
    </div>
  </div>
</div>

<script>
const form = document.getElementById("uploadForm");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const countDisplay = document.getElementById("count");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  loading.style.display = "block";
  result.style.display = "none";

  const file = document.getElementById("imageInput").files[0];
  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch("/detect", { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    countDisplay.textContent = data.count;
    document.getElementById("originalImage").src = URL.createObjectURL(file);
    document.getElementById("densityMap").src = `data:image/jpeg;base64,${data.density_map}`;
    document.getElementById("overlayImage").src = `data:image/jpeg;base64,${data.overlay}`;
    result.style.display = "block";
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    loading.style.display = "none";
  }
});
</script>

</body>
</html>
"""

# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    count, density_map_b64, overlay_b64 = predict_and_visualize(filepath)

    return jsonify({
        "count": count,
        "density_map": density_map_b64,
        "overlay": overlay_b64
    })

if __name__ == "__main__":
    app.run(debug=True)
