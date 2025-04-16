import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# === Configuration ===
INPUT_IMAGE = 'imgs/image1.jpeg'  # Grayscale input image
OUTPUT_FOLDER = 'imgs_out'
MODEL_FOLDER = 'models'

# === Ensure output folder exists ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Load model files ===
prototxt = os.path.join(MODEL_FOLDER, 'colorization_deploy_v2.prototxt')
model = os.path.join(MODEL_FOLDER, 'colorization_release_v2.caffemodel')
points = os.path.join(MODEL_FOLDER, 'pts_in_hull.npy')

# === Load pre-trained Caffe model ===
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)
pts = pts.transpose().reshape(2, 313, 1, 1)

# === Inject cluster centers into the model ===
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# === Read the grayscale image ===
frame = cv2.imread(INPUT_IMAGE)
if frame is None:
    raise ValueError(f"❌ Could not read image at path: {INPUT_IMAGE}")

# === Preprocessing ===
scaled = frame.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
resized = cv2.resize(lab, (224, 224))
L = resized[:, :, 0]
L -= 50  # mean-centering

# === Predict a and b channels ===
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))

# === Reconstruct colorized image ===
L_orig = lab[:, :, 0]
colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

# === Save colorized image ===
output_path = os.path.join(OUTPUT_FOLDER, 'image1_output.jpeg')
cv2.imwrite(output_path, (colorized * 255).astype("uint8"))
print(f"Colorized image saved at: {output_path}")

# === Display images (resized for screen) ===
def show_resized(title, img, max_dim=600):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(title, resized)

show_resized("Original", frame)
show_resized("Colorized", (colorized * 255).astype("uint8"))

cv2.waitKey(0)
cv2.destroyAllWindows()

# === Evaluation ===
ground_truth_path = 'imgs/original.jpeg'  
ground_truth = cv2.imread(ground_truth_path)

if ground_truth is None:
    raise ValueError(f"❌ Could not read ground truth image: {ground_truth_path}")

# Resize ground truth to match colorized output
ground_truth = cv2.resize(ground_truth, (colorized.shape[1], colorized.shape[0]))

# === Debug shapes ===
print("✅ Ground truth shape:", ground_truth.shape)
print("✅ Predicted image shape:", colorized.shape)

# Convert both to same format for metric calculation
pred = (colorized * 255).astype("uint8")

# === PSNR and SSIM ===
try:
    psnr_value = psnr(ground_truth, pred)
    ssim_value = ssim(ground_truth, pred, channel_axis=-1)  # use channel_axis instead of multichannel
    print(f"✅ PSNR: {psnr_value:.2f} dB")
    print(f"✅ SSIM: {ssim_value:.4f}")
except Exception as e:
    print("❌ Error during metric evaluation:", str(e))
