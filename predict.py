import torch
from model import CSRNet
import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with weights_only=False for PyTorch 2.6+ compatibility
model = CSRNet().to(device)
try:
    checkpoint = torch.load("PartAmodel_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()

# Enhanced image loading and preprocessing
def load_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize for better accuracy
        img_resized = cv2.resize(img, (1024, 768))  # (width, height)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(img_resized).unsqueeze(0), img_resized
    except Exception as e:
        print(f"Image loading error: {e}")
        exit()

# Prediction and visualization
def predict_and_visualize(image_path):
    # Load and process image
    img_tensor, orig_img = load_image(image_path)
    img_tensor = img_tensor.to(device)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)

    # Process and enhance density map
    density_map = output.squeeze().cpu().numpy()
    density_map = gaussian_filter(density_map, sigma=1)  # Smoothen
    density_map = np.clip(density_map, 0, None)  # Remove negatives
    count = int(round(np.sum(density_map)))  # Round to nearest integer

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img)
    plt.title("Original Image")
    plt.axis('off')

    # Density map
    plt.subplot(1, 3, 2)
    plt.imshow(density_map, cmap='jet')
    plt.title(f"Density Map\nEstimated Count: {count}")
    plt.colorbar()
    plt.axis('off')

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(orig_img)
    plt.imshow(density_map, cmap='jet', alpha=0.5)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run prediction
if __name__ == "__main__":
    image_path = "crowded.jpg"  # Replace with your image path
    predict_and_visualize(image_path)
