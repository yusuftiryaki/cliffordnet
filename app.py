import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os
from escnn import gspaces
from escnn import nn as gnn

# =========================================================================
# MODEL TANIMLARI (Notebook ile Uyumlu)
# =========================================================================

class CliffordBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # D12 Grubu: 30 derece dönüşler + Yansımalar (12'lik rotasyon)
        self.r2_act = gspaces.flipRot2dOnR2(N=12)
        self.in_type = gnn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # Temsil Tanımları
        scalar_repr = self.r2_act.trivial_repr
        vector_repr = self.r2_act.irrep(1, 1)   # 2D
        bivector_repr = self.r2_act.irrep(1, 0) # 1D

        # Özellik Listesi: 20 * 2 + 4 * 1 = 44 Kanal
        geo_feature_list = (
            20 * [vector_repr] +
            4 * [bivector_repr]
        )
        self.hidden_type = gnn.FieldType(self.r2_act, geo_feature_list)

        # --- Bloklar ---
        self.block1 = nn.Sequential(
            gnn.R2Conv(self.in_type, self.hidden_type, kernel_size=3, padding=1, bias=False),
            gnn.GNormBatchNorm(self.hidden_type),
            gnn.NormNonLinearity(self.hidden_type),
        )
        self.block2 = nn.Sequential(
            gnn.R2Conv(self.hidden_type, self.hidden_type, kernel_size=3, padding=1, bias=False),
            gnn.GNormBatchNorm(self.hidden_type),
            gnn.NormNonLinearity(self.hidden_type),
        )
        self.block3 = nn.Sequential(
            gnn.R2Conv(self.hidden_type, self.hidden_type, kernel_size=3, padding=1, bias=False),
            gnn.GNormBatchNorm(self.hidden_type),
            gnn.NormNonLinearity(self.hidden_type),
        )

        # --- Spatial Attention ---
        self.att_type = gnn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.att_conv = gnn.R2Conv(self.hidden_type, self.att_type, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        # --- Spatial Pyramid Pooling (SPP) ---
        # Levels: 1x1, 2x2, 4x4
        self.spp_levels = [1, 2, 4]
        # Output channels calculation
        self.out_channels = self.hidden_type.size * sum([l*l for l in self.spp_levels])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = gnn.GeometricTensor(x, self.in_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Spatial Attention Calculation
        att_map = self.att_conv(x)             
        att_map = self.sigmoid(att_map.tensor) 
        
        # Apply Attention
        out = x.tensor * att_map
        
        # Apply SPP
        spp_outputs = []
        for level in self.spp_levels:
            pooled = nn.functional.adaptive_avg_pool2d(out, output_size=level)
            spp_outputs.append(pooled.view(pooled.size(0), -1))
            
        return torch.cat(spp_outputs, dim=1)

class ResNetBackbone(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # Load ResNet18
        resnet = models.resnet18(weights=None) 
        
        # Modify first layer if input is not 3 channels (RGB)
        if in_channels != 3:
            original_conv1 = resnet.conv1
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # Remove FC layer to get features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.out_channels = 512
        
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1) # Flatten (Batch, 512)

class ResNetClassifier(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.out_channels, n_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        return out

class HybridClassifier(nn.Module):
    def __init__(self, clifford_backbone, resnet_backbone, n_classes):
        super().__init__()
        self.clifford_backbone = clifford_backbone
        self.resnet_backbone = resnet_backbone
        
        # Concatenate features
        self.in_features = clifford_backbone.out_channels + resnet_backbone.out_channels
        self.fc = nn.Linear(self.in_features, n_classes)
        
    def forward(self, x):
        # Clifford Path (Needs Grayscale)
        # If input is RGB (3 channels), convert to Grayscale
        if x.shape[1] == 3:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x
            
        c_feat = self.clifford_backbone(x_gray) 
        
        # ResNet Path (Takes x directly)
        r_feat = self.resnet_backbone(x) 
        
        # Concatenate
        combined = torch.cat((c_feat, r_feat), dim=1)
        
        out = self.fc(combined)
        return out

# =========================================================================
# UYGULAMA MANTIĞI
# =========================================================================

device = torch.device("cpu") # Demo için CPU yeterli
n_classes = 5 # RetinaMNIST (0-4)
labels = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"} # RetinaMNIST labels

# --- Model 1: ResNet Only ---
resnet_backbone_only = ResNetBackbone(in_channels=3)
model_resnet = ResNetClassifier(resnet_backbone_only, n_classes)
RESNET_PATH = "resnet_retinamnist_model.pth"

if os.path.exists(RESNET_PATH):
    try:
        model_resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device))
        print("✅ ResNet model ağırlıkları yüklendi.")
    except Exception as e:
        print(f"❌ ResNet model yüklenirken hata: {e}")
else:
    print(f"⚠️ ResNet model dosyası ({RESNET_PATH}) bulunamadı.")

model_resnet.to(device)
model_resnet.eval()

# --- Model 2: Hybrid ---
clifford_backbone = CliffordBackbone()
resnet_backbone_hybrid = ResNetBackbone(in_channels=3) # RGB için 3 kanal
model_hybrid = HybridClassifier(clifford_backbone, resnet_backbone_hybrid, n_classes)
HYBRID_PATH = "hybrid_retinamnist_model.pth"

if os.path.exists(HYBRID_PATH):
    try:
        model_hybrid.load_state_dict(torch.load(HYBRID_PATH, map_location=device))
        print("✅ Hybrid model ağırlıkları yüklendi.")
    except Exception as e:
        print(f"❌ Hybrid model yüklenirken hata: {e}")
else:
    print(f"⚠️ Hybrid model dosyası ({HYBRID_PATH}) bulunamadı.")

model_hybrid.to(device)
model_hybrid.eval()

# 2. Görüntü İşleme (RGB)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # RGB Normalizasyon
])

def predict_both(image):
    if image is None:
        return None, None
    
    # PIL Image -> Tensor
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # ResNet Prediction
    with torch.no_grad():
        out_r = model_resnet(img_tensor)
        prob_r = torch.nn.functional.softmax(out_r, dim=1)[0]
        res_r = {labels[i]: float(prob_r[i]) for i in range(len(labels))}
        
    # Hybrid Prediction
    with torch.no_grad():
        out_h = model_hybrid(img_tensor)
        prob_h = torch.nn.functional.softmax(out_h, dim=1)[0]
        res_h = {labels[i]: float(prob_h[i]) for i in range(len(labels))}
        
    return res_r, res_h

# 3. Gradio Arayüzü
with gr.Blocks(title="RetinaMNIST Model Comparison") as demo:
    gr.Markdown("# 👁️ RetinaMNIST: ResNet vs Hybrid (Clifford) Comparison")
    gr.Markdown("Bu arayüz, klasik ResNet18 ile Geometrik (Clifford) özelliklerle güçlendirilmiş Hybrid modelin tahminlerini karşılaştırır.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Retina Görüntüsü Yükleyin")
            btn = gr.Button("Tahmin Et", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 🧠 Klasik ResNet18")
            out_resnet = gr.Label(num_top_classes=5, label="ResNet Tahmini")
            
        with gr.Column():
            gr.Markdown("### 🧬 Hybrid (Clifford + ResNet)")
            out_hybrid = gr.Label(num_top_classes=5, label="Hybrid Tahmini")
            
    btn.click(fn=predict_both, inputs=input_img, outputs=[out_resnet, out_hybrid])

if __name__ == "__main__":
    demo.launch()
