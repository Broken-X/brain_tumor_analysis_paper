import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ==========================================
# 1. MODEL & UTILS
# ==========================================

def build_classifier_head(in_features: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

def get_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower().strip()
    if name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = build_classifier_head(in_features, num_classes)
        return model

    if name == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = build_classifier_head(in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = build_classifier_head(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {name}")

def get_target_layer(model_name: str, model: nn.Module) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "densenet121":
        return model.features.denseblock4
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No target layer mapping for {model_name}")

# ==========================================
# 2. GRAD-CAM CLASS
# ==========================================

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def fwd_hook(_m, _inp, out):
            self.activations = out

        def bwd_hook(_m, _gin, gout):
            self.gradients = gout[0]

        self.h1 = self.target_layer.register_forward_hook(fwd_hook)
        self.h2 = self.target_layer.register_full_backward_hook(bwd_hook)

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        A = self.activations
        G = self.gradients
        weights = torch.mean(G, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * A, dim=1)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()[0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

def overlay_cam(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img = np.array(img_pil).astype(np.float32) / 255.0
    H, W = img.shape[:2]
    
    cam_t = torch.tensor(cam, dtype=torch.float32)[None, None, :, :]
    cam_up = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    
    cmap = plt.get_cmap("jet")
    heat = cmap(cam_up)[:, :, :3]
    
    out = (1 - alpha) * img + alpha * heat
    out = np.clip(out, 0, 1)
    return out

# ==========================================
# 3. STREAMLIT APP LOGIC
# ==========================================

st.set_page_config(page_title="Brain Tumor AI Multi-Model", layout="wide")

st.title("🧠 Multi-Model Brain Tumor Analysis")
st.markdown("""
Compare how different models (**DenseNet, EfficientNet, ResNet**) classify and visualize the same MRI scans.
""")

# --- Sidebar: Model Selection ---
st.sidebar.header("Configuration")
results_dir = Path("results/checkpoints")

if not results_dir.exists():
    st.error(f"Checkpoint directory not found at {results_dir}")
    st.stop()

# Find available .pth files
checkpoints = list(results_dir.glob("*.pth"))
ckpt_names = [p.name for p in checkpoints]

if not ckpt_names:
    st.error("No checkpoints found.")
    st.stop()

# Allow multiple model selection
selected_ckpt_names = st.sidebar.multiselect(
    "Select Models to Compare", 
    ckpt_names,
    default=[ckpt_names[0]] if ckpt_names else None
)

if not selected_ckpt_names:
    st.warning("Please select at least one model from the sidebar.")
    st.stop()

# --- Load Model Function (Cached) ---
@st.cache_resource
def load_prediction_model(ckpt_path):
    device = torch.device("cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    model_name = checkpoint.get("model_name", "densenet121")
    class_names = checkpoint.get("class_names", ["glioma", "meningioma", "no_tumor", "pituitary"])
    img_size = checkpoint.get("img_size", 224)
    
    model = get_model(model_name, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    return model, class_names, img_size, model_name, device

# --- Preprocessing ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Main Interface ---

# 1. Choose Input Source
input_source = st.radio("Select Image Source:", ["Upload Image", "Sample Directory"])

process_queue = [] # List of tuples: (filename, PIL Image)

if input_source == "Upload Image":
    uploaded_files = st.file_uploader(
        "Upload MRI Images", 
        type=["jpg", "png", "jpeg", "tif"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        for uf in uploaded_files:
            img = Image.open(uf).convert("RGB")
            process_queue.append((uf.name, img))

else: # Sample Directory
    
    # NEW: Dictionary mapping display names to folder paths
    DATASET_OPTIONS = {
        "Physician Review Sample": Path("results/physician_review_sample"),
        "Masoud Nickparvar Dataset": Path("datasets/masoudnickparvar/brain-tumor-mri-dataset"),
        "PK Darabi Dataset": Path("datasets/pkdarabi/medical-image-dataset-brain-tumor-detection-organized")
    }
    
    # Dropdown to select which folder to browse
    selected_dataset_name = st.selectbox("Select Dataset Folder", list(DATASET_OPTIONS.keys()))
    sample_dir = DATASET_OPTIONS[selected_dataset_name]

    if not sample_dir.exists():
        st.error(f"Directory not found: {sample_dir}")
        st.info("Please ensure the folder exists in the project root.")
    else:
        # Recursive search: Look in all subfolders (glioma, pituitary, etc.)
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        
        # Gather all image files from subdirectories
        # rglob("*") finds all files recursively
        sample_files = [
            str(f.relative_to(sample_dir)) 
            for f in sample_dir.rglob("*") 
            if f.suffix.lower() in valid_exts
        ]
        
        # Sort them for easier finding
        sample_files.sort()

        if not sample_files:
            st.warning("No images found in this directory or its subfolders.")
        else:
            selected_samples = st.multiselect("Select Sample Images", sample_files)
            for sample_rel_path in selected_samples:
                file_path = sample_dir / sample_rel_path
                try:
                    img = Image.open(file_path).convert("RGB")
                    process_queue.append((sample_rel_path, img))
                except Exception as e:
                    st.error(f"Error loading {sample_rel_path}: {e}")

# 2. Process Images
if process_queue:
    for filename, image in process_queue:
        st.divider()
        st.subheader(f"📄 File: {filename}")
        
        # Create columns: 1 for Original + N for Selected Models
        cols = st.columns(1 + len(selected_ckpt_names))
        
        # Column 0: Original Image
        with cols[0]:
            st.markdown("**Original**")
            st.image(image, use_container_width=True)

        # Loop through selected models
        for idx, ckpt_name in enumerate(selected_ckpt_names):
            ckpt_path = results_dir / ckpt_name
            
            try:
                # Load specific model
                model, class_names, img_size, model_name, device = load_prediction_model(ckpt_path)
                
                # Transform image for this model
                tfms = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
                input_tensor = tfms(image).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    pred_idx = probs.argmax()
                    pred_class = class_names[pred_idx]
                    conf = probs[pred_idx]

                # Grad-CAM
                target_layer = get_target_layer(model_name, model)
                cammer = GradCAM(model, target_layer)
                cam_map = cammer.generate(input_tensor, pred_idx)
                cammer.close()
                
                img_resized = image.resize((img_size, img_size))
                overlay = overlay_cam(img_resized, cam_map, alpha=0.5)

                # Display in the specific column
                with cols[idx + 1]:
                    st.markdown(f"**{ckpt_name}**") 
                    st.image(overlay, caption=f"Pred: {pred_class.upper()}", use_container_width=True)
                    
                    st.markdown(f"**Confidence: {conf*100:.2f}%**")
                    st.markdown("---")
                    st.markdown("All Probabilities:")
                    for c_name, c_prob in zip(class_names, probs):
                        # Highlight the predicted class
                        if c_name == pred_class:
                            st.markdown(f"**• {c_name}: {c_prob*100:.1f}%**")
                        else:
                            st.markdown(f"• {c_name}: {c_prob*100:.1f}%")

            except Exception as e:
                with cols[idx + 1]:
                    st.error(f"Error: {e}")

else:
    st.info("Select or upload images to begin analysis.")