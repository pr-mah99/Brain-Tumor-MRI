import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import os
import cv2
from pathlib import Path

# ======================== Model Components ========================
class MicroAdaptiveFeatureExtractor(nn.Module):
    """
    MAFE - Parallel version (Eq. 3.2-3.4)
    Spatial and channel attention are both computed directly from x
    (not sequentially), then combined multiplicatively.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_ch = max(1, channels // reduction)
        self.spatial_conv = nn.Conv2d(channels, 1, 1)
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Conv2d(channels, reduced_ch, 1)
        self.channel_out = nn.Conv2d(reduced_ch, channels, 1)
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad=True)

    def forward(self, x):
        # Both branches computed independently from the original x (parallel)
        spatial_weights = torch.sigmoid(self.spatial_conv(x))                 # Ws from x

        channel_pool = self.channel_avg(x)                                     # from x, not x_spatial
        channel_weights = torch.sigmoid(self.channel_out(F.relu(self.channel_fc(channel_pool))))  # Wc from x

        x_enhanced = x * spatial_weights * channel_weights                     # x · Ws · Wc combined

        return x + self.alpha * x_enhanced


class MicroMultiScaleProcessor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        inter = max(8, channels // 4)
        while inter % 4 != 0 and inter > 4:
            inter -= 1
        if inter < 4:
            inter = 4

        self.scale1 = nn.Conv2d(channels, inter, 1, bias=False)
        self.scale3 = nn.Conv2d(channels, inter, 3, padding=1, bias=False)

        if channels >= 8 and inter >= 4:
            groups = 1
            for g in [8, 4, 2]:
                if channels % g == 0:
                    groups = g
                    break
            self.scale3dw = nn.Conv2d(channels, inter, 3, padding=1, groups=groups, bias=False)
        else:
            self.scale3dw = nn.Conv2d(channels, inter, 3, padding=1, bias=False)

        self.fusion = nn.Conv2d(inter * 3, channels, 1, bias=False)
        self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)

    def forward(self, x):
        s1 = F.relu(self.scale1(x))
        s3 = F.relu(self.scale3(x))
        s3dw = F.relu(self.scale3dw(x))
        combined = torch.cat([s1, s3, s3dw], dim=1)
        fused = self.fusion(combined)
        return F.relu(x + self.beta * fused)


class MicroIntelligentPooling(nn.Module):
    """
    CAIP - Context-Aware Intelligent Pooling (Eq. 3.9-3.12)
    A spatial context map (same H×W as input) reweights x before pooling,
    so the pooling decision is context-aware rather than a single global
    scalar decision per image.
    """
    def __init__(self, channels):
        super().__init__()
        reduced_ch = max(4, channels // 4)

        # Context Analysis network - preserves spatial dimensions
        self.context_net = nn.Sequential(
            nn.Conv2d(channels, reduced_ch, 1),
            nn.BatchNorm2d(reduced_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, reduced_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, channels, 3, padding=1),
            nn.Sigmoid()
        )

        self.lam = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # learnable λ

    def forward(self, x):
        context = self.context_net(x)          # context map, same shape as x
        x_weighted = context * x                # Context ⊗ x

        y_max = F.max_pool2d(x_weighted, kernel_size=2, stride=2)
        y_avg = F.avg_pool2d(x_weighted, kernel_size=2, stride=2)

        lam_clamped = torch.sigmoid(self.lam)   # constrain λ to [0, 1]
        return lam_clamped * y_max + (1 - lam_clamped) * y_avg


class UltraLightPatentBrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4, base_channels=[32, 64, 128, 256]):
        super().__init__()
        c1, c2, c3, c4 = base_channels

        self.conv1 = nn.Conv2d(3, c1, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.adaptive1 = MicroAdaptiveFeatureExtractor(c1, reduction=4)
        self.pool1 = MicroIntelligentPooling(c1)

        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.multiscale2 = MicroMultiScaleProcessor(c2)
        self.pool2 = MicroIntelligentPooling(c2)

        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c3)
        self.adaptive3 = MicroAdaptiveFeatureExtractor(c3, reduction=8)
        self.pool3 = MicroIntelligentPooling(c3)

        self.conv4 = nn.Conv2d(c3, c4, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(c4)
        self.multiscale4 = MicroMultiScaleProcessor(c4)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(c4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.adaptive1(x)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.multiscale2(x)
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive3(x)
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.multiscale4(x)

        return self.classifier(x)


# ======================== Grad-CAM ========================
class GradCAM:
    """
    Grad-CAM: يلتقط التنشيطات والتدرجات من الطبقة المستهدفة (conv4)
    لإنتاج خريطة حرارية توضح المناطق المؤثرة في قرار التصنيف.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        # تمكين الـ gradient مؤقتاً
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # حساب أوزان القنوات (global average pooling على التدرجات)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # الجمع الموزون للخرائط التنشيطية
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # تطبيع بين 0 و 1
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        probabilities = F.softmax(output, dim=1)[0].detach().cpu().numpy()

        return cam, class_idx, probabilities


def apply_gradcam_overlay(original_img_pil, cam, alpha=0.45):
    """
    دمج خريطة Grad-CAM الحرارية مع الصورة الأصلية.
    - original_img_pil: PIL Image (RGB)
    - cam: numpy array بين 0 و 1
    - alpha: شفافية خريطة الحرارة
    """
    # تحويل الصورة إلى numpy BGR لـ OpenCV
    orig_np = np.array(original_img_pil)  # RGB uint8
    h, w = orig_np.shape[:2]

    # تكبير خريطة Grad-CAM لتطابق أبعاد الصورة
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    # تحويل إلى خريطة ألوان (JET colormap)
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # دمج الصورة الأصلية مع الخريطة الحرارية
    overlay = (orig_np.astype(np.float32) * (1 - alpha) +
               heatmap_rgb.astype(np.float32) * alpha).astype(np.uint8)

    # إضافة حدود بيضاء حول النقاط عالية التنشيط (threshold > 0.7)
    mask = (cam_resized > 0.70).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (255, 255, 255), 1)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # تحويل النتيجة لـ base64
    result_pil = Image.fromarray(overlay)
    buf = BytesIO()
    result_pil.save(buf, format='JPEG', quality=92)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def pil_to_base64(img_pil):
    buf = BytesIO()
    img_pil.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ======================== Flask App ========================
app = Flask(__name__)
device = torch.device('cpu')

model = UltraLightPatentBrainTumorCNN(num_classes=4).to(device)
checkpoint = torch.load('best_brain_tumor_model_patent.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])

# تهيئة Grad-CAM على الطبقة conv4 (آخر طبقة تلافيفية = أعمق تمثيل)
gradcam = GradCAM(model, target_layer=model.conv4)

# ======================== تحميل الصور من المجلد ========================
def load_sample_images():
    sample_dir = Path('sample_images')
    sample_images = {}
    for class_name in class_names:
        class_dir = sample_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.JPG'))
            sample_images[class_name] = images[:20]
    return sample_images

sample_images = load_sample_images()

def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


# ======================== HTML Template ========================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تصنيف أورام الدماغ + Grad-CAM</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px 16px;
        }

        .container {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            backdrop-filter: blur(20px);
            border-radius: 28px;
            box-shadow: 0 32px 64px rgba(0,0,0,0.5);
            max-width: 960px;
            width: 100%;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 36px 32px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 60% 40%, rgba(255,255,255,0.08) 0%, transparent 60%);
            pointer-events: none;
        }

        .header h1 {
            font-size: 30px;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 6px;
        }

        .header p {
            font-size: 13px;
            opacity: 0.85;
            letter-spacing: 0.5px;
        }

        .badge {
            display: inline-block;
            margin-top: 12px;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .content { padding: 32px; }

        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 28px;
        }

        .tab-btn {
            flex: 1;
            padding: 11px 16px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            color: rgba(255,255,255,0.5);
            transition: all 0.25s;
        }

        .tab-btn.active {
            background: linear-gradient(135deg, rgba(102,126,234,0.35), rgba(118,75,162,0.35));
            border-color: rgba(102,126,234,0.6);
            color: white;
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .upload-zone {
            border: 2px dashed rgba(102,126,234,0.5);
            border-radius: 18px;
            padding: 40px 30px;
            text-align: center;
            cursor: pointer;
            background: rgba(102,126,234,0.06);
            transition: all 0.3s ease;
        }

        .upload-zone:hover {
            border-color: rgba(102,126,234,0.9);
            background: rgba(102,126,234,0.12);
            transform: translateY(-2px);
        }

        .upload-zone input { display: none; }
        .upload-icon { font-size: 52px; margin-bottom: 12px; }
        .upload-text { color: #a5b4fc; font-weight: 700; font-size: 16px; margin-bottom: 4px; }
        .upload-subtext { color: rgba(255,255,255,0.35); font-size: 12px; }

        .samples-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }

        .sample-item {
            cursor: pointer;
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid transparent;
            transition: all 0.3s;
        }

        .sample-item:hover {
            border-color: #667eea;
            transform: translateY(-5px);
            box-shadow: 0 10px 24px rgba(102,126,234,0.4);
        }

        .sample-item img {
            width: 100%;
            height: 110px;
            object-fit: cover;
            display: block;
        }

        .preview-section {
            display: none;
            margin: 24px 0 0;
        }

        /* ── مقارنة جنباً إلى جنب ── */
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }

        .img-card {
            border-radius: 16px;
            overflow: hidden;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .img-card-label {
            padding: 10px 14px;
            font-size: 12px;
            font-weight: 700;
            color: rgba(255,255,255,0.6);
            letter-spacing: 0.8px;
            text-transform: uppercase;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .label-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .dot-original { background: #60a5fa; }
        .dot-gradcam  { background: #f97316; }

        .img-card img {
            width: 100%;
            height: 240px;
            object-fit: contain;
            display: block;
            background: #0a0a0a;
        }

        /* placeholders الشبح */
        .img-placeholder {
            width: 100%;
            height: 240px;
            background: rgba(255,255,255,0.03);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: rgba(255,255,255,0.2);
            font-size: 32px;
            gap: 8px;
        }

        .img-placeholder span { font-size: 11px; letter-spacing: 0.5px; }

        .controls {
            display: flex;
            gap: 12px;
            margin-bottom: 28px;
        }

        button {
            flex: 1;
            padding: 14px;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            cursor: pointer;
            font-size: 15px;
            transition: all 0.25s;
        }

        .btn-classify {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 6px 20px rgba(102,126,234,0.35);
        }

        .btn-classify:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(102,126,234,0.5);
        }

        .btn-classify:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-clear {
            flex: 0.4;
            background: rgba(255,255,255,0.07);
            color: rgba(255,255,255,0.6);
            border: 1px solid rgba(255,255,255,0.12);
        }

        .btn-clear:hover { background: rgba(255,255,255,0.12); color: white; }

        /* ── Loading ── */
        .loading {
            display: none;
            text-align: center;
            padding: 32px;
        }

        .scanner-ring {
            width: 60px;
            height: 60px;
            border: 3px solid rgba(102,126,234,0.2);
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 14px;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .loading-text {
            color: rgba(255,255,255,0.55);
            font-size: 14px;
            animation: fade 1.2s ease-in-out infinite;
        }

        @keyframes fade { 0%,100% { opacity:0.4; } 50% { opacity:1; } }

        /* ── Results ── */
        .results {
            display: none;
            margin-top: 8px;
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(135deg, rgba(102,126,234,0.18), rgba(118,75,162,0.18));
            border: 1px solid rgba(102,126,234,0.35);
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 16px;
        }

        .result-label {
            font-size: 11px;
            font-weight: 700;
            color: rgba(255,255,255,0.45);
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .result-value {
            font-size: 30px;
            font-weight: 800;
            color: #a5b4fc;
        }

        .confidence-pill {
            background: rgba(102,126,234,0.25);
            border: 1px solid rgba(102,126,234,0.45);
            border-radius: 20px;
            padding: 8px 20px;
            font-size: 22px;
            font-weight: 800;
            color: white;
        }

        .scores-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 4px;
        }

        .score-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 12px 16px;
            transition: all 0.2s;
        }

        .score-card.top-class {
            background: rgba(102,126,234,0.15);
            border-color: rgba(102,126,234,0.4);
        }

        .score-name {
            font-size: 12px;
            color: rgba(255,255,255,0.5);
            margin-bottom: 8px;
            font-weight: 600;
        }

        .score-bar-bg {
            height: 6px;
            background: rgba(255,255,255,0.08);
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 6px;
        }

        .score-bar-fill {
            height: 100%;
            border-radius: 6px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.6s cubic-bezier(0.22,1,0.36,1);
        }

        .score-pct {
            font-size: 18px;
            font-weight: 800;
            color: white;
        }

        /* Grad-CAM legend */
        .gradcam-legend {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            background: rgba(0,0,0,0.25);
            border-top: 1px solid rgba(255,255,255,0.06);
            font-size: 11px;
            color: rgba(255,255,255,0.45);
            flex-wrap: wrap;
        }

        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-swatch {
            width: 22px;
            height: 8px;
            border-radius: 4px;
        }

        .error {
            display: none;
            background: rgba(220,38,38,0.15);
            color: #fca5a5;
            padding: 14px 18px;
            border-radius: 12px;
            border-left: 4px solid #ef4444;
            margin-top: 16px;
        }

        .section-title {
            color: rgba(255,255,255,0.55);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 14px;
        }

        .sample-class-title {
            color: #a5b4fc;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(102,126,234,0.2);
        }

        .scroll-up {
            position: fixed;
            bottom: 28px;
            right: 28px;
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            font-size: 20px;
            display: none;
            align-items: center;
            justify-content: center;
            box-shadow: 0 6px 20px rgba(102,126,234,0.5);
            transition: all 0.3s;
            z-index: 999;
            flex: 0;
            padding: 0;
        }

        .scroll-up:hover { transform: translateY(-4px); }

        @media (max-width: 600px) {
            .comparison-grid { grid-template-columns: 1fr; }
            .scores-grid { grid-template-columns: 1fr; }
            .header h1 { font-size: 22px; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🧠 تصنيف أورام الدماغ</h1>
        <p>Brain Tumor Classification — Deep Learning CNN</p>
        <div class="badge">✦ Grad-CAM Visualization</div>
    </div>

    <div class="content">
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('upload', this)">📤 رفع صورة</button>
            <button class="tab-btn" onclick="switchTab('samples', this)">🗂 صور عينات</button>
        </div>

        <!-- تبويب الرفع -->
        <div class="tab-content active" id="tab-upload">
            <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*">
                <div class="upload-icon">📤</div>
                <div class="upload-text">اضغط لرفع صورة MRI</div>
                <div class="upload-subtext">PNG · JPG · WEBP مقبول</div>
            </div>
        </div>

        <!-- تبويب العينات -->
        <div class="tab-content" id="tab-samples">
            <div id="samplesContainer"></div>
        </div>

        <!-- منطقة المعاينة + التحكم (مشتركة) -->
        <div class="preview-section" id="previewSection">

            <div class="comparison-grid">
                <!-- الصورة الأصلية -->
                <div class="img-card">
                    <div class="img-card-label">
                        <span class="label-dot dot-original"></span>
                        الصورة الأصلية
                    </div>
                    <img id="originalImg" src="" alt="Original">
                </div>

                <!-- Grad-CAM -->
                <div class="img-card">
                    <div class="img-card-label">
                        <span class="label-dot dot-gradcam"></span>
                        Grad-CAM — منطقة الورم
                    </div>
                    <div class="img-placeholder" id="gradcamPlaceholder">
                        🔬 <span>يظهر بعد التصنيف</span>
                    </div>
                    <img id="gradcamImg" src="" alt="Grad-CAM" style="display:none">
                </div>
            </div>

            <!-- legend خريطة الحرارة -->
            <div class="gradcam-legend">
                <span>شدة التنشيط:</span>
                <div class="legend-item">
                    <div class="legend-swatch" style="background:linear-gradient(90deg,#00008b,#0000ff)"></div>
                    <span>منخفض</span>
                </div>
                <div class="legend-item">
                    <div class="legend-swatch" style="background:linear-gradient(90deg,#008000,#ffff00)"></div>
                    <span>متوسط</span>
                </div>
                <div class="legend-item">
                    <div class="legend-swatch" style="background:linear-gradient(90deg,#ffa500,#ff0000)"></div>
                    <span>عالي (منطقة الورم)</span>
                </div>
                <span style="margin-right:auto; color:rgba(255,255,255,0.3)">الحدود البيضاء = تنشيط > 70%</span>
            </div>

            <div class="controls" style="margin-top:16px">
                <button class="btn-classify" id="classifyBtn" onclick="classify()">🔍 تصنيف + Grad-CAM</button>
                <button class="btn-clear" onclick="clearAll()">✕ مسح</button>
            </div>
        </div>

        <!-- Loading -->
        <div class="loading" id="loading">
            <div class="scanner-ring"></div>
            <div class="loading-text">جاري التحليل وحساب Grad-CAM...</div>
        </div>

        <!-- النتائج -->
        <div class="results" id="results">
            <div class="result-header">
                <div>
                    <div class="result-label">التشخيص</div>
                    <div class="result-value" id="resultValue">—</div>
                </div>
                <div class="confidence-pill" id="confPill">—%</div>
            </div>

            <div class="section-title">توزيع الاحتمالات</div>
            <div class="scores-grid" id="scoresGrid"></div>
        </div>

        <div class="error" id="error"></div>
    </div>
</div>

<button class="scroll-up" id="scrollUp" onclick="window.scrollTo({top:0,behavior:'smooth'})">⬆</button>

<script>
    let selectedFile = null;

    // ── Tabs ──
    function switchTab(name, btn) {
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('tab-' + name).classList.add('active');
        btn.classList.add('active');
    }

    // ── File input ──
    document.getElementById('fileInput').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = ev => showPreview(ev.target.result, null);
        reader.readAsDataURL(file);
    });

    function showPreview(src, gradcamSrc) {
        document.getElementById('originalImg').src = src;
        document.getElementById('previewSection').style.display = 'block';

        if (gradcamSrc) {
            document.getElementById('gradcamImg').src = 'data:image/jpeg;base64,' + gradcamSrc;
            document.getElementById('gradcamImg').style.display = 'block';
            document.getElementById('gradcamPlaceholder').style.display = 'none';
        } else {
            document.getElementById('gradcamImg').style.display = 'none';
            document.getElementById('gradcamPlaceholder').style.display = 'flex';
        }
    }

    // ── Sample images ──
    function loadSampleImages(data) {
        const container = document.getElementById('samplesContainer');
        container.innerHTML = '';
        data.samples.forEach(cat => {
            const wrap = document.createElement('div');
            wrap.style.marginBottom = '28px';

            const title = document.createElement('div');
            title.className = 'sample-class-title';
            title.textContent = cat.class_name;
            wrap.appendChild(title);

            const grid = document.createElement('div');
            grid.className = 'samples-grid';

            cat.images.forEach(img => {
                const item = document.createElement('div');
                item.className = 'sample-item';
                item.innerHTML = '<img src="' + img.src + '" alt="sample" loading="lazy">';
                item.onclick = () => selectSample(img.path, img.src);
                grid.appendChild(item);
            });

            wrap.appendChild(grid);
            container.appendChild(wrap);
        });
    }

    function selectSample(path, previewSrc) {
        selectedFile = { path };
        showPreview(previewSrc, null);
        document.getElementById('results').style.display = 'none';
        // انتقل لتبويب الرفع وأرِ الصورة
        switchTab('upload', document.querySelector('.tab-btn'));
        setTimeout(() => document.getElementById('previewSection').scrollIntoView({behavior:'smooth'}), 100);
    }

    // ── Classify ──
    async function classify() {
        if (!selectedFile) return;

        const btn = document.getElementById('classifyBtn');
        btn.disabled = true;

        // إخفاء النتائج السابقة
        document.getElementById('results').style.display = 'none';
        document.getElementById('error').style.display = 'none';
        document.getElementById('loading').style.display = 'block';

        // إعادة placeholder Grad-CAM
        document.getElementById('gradcamImg').style.display = 'none';
        document.getElementById('gradcamPlaceholder').style.display = 'flex';

        const formData = new FormData();
        if (selectedFile.path) {
            formData.append('sample_path', selectedFile.path);
        } else {
            formData.append('image', selectedFile);
        }

        try {
            const res = await fetch('/predict', { method: 'POST', body: formData });
            const data = await res.json();
            document.getElementById('loading').style.display = 'none';

            if (data.error) {
                document.getElementById('error').textContent = 'خطأ: ' + data.error;
                document.getElementById('error').style.display = 'block';
            } else {
                // ── عرض Grad-CAM ──
                document.getElementById('gradcamImg').src = 'data:image/jpeg;base64,' + data.gradcam_image;
                document.getElementById('gradcamImg').style.display = 'block';
                document.getElementById('gradcamPlaceholder').style.display = 'none';

                // ── النتائج ──
                document.getElementById('resultValue').textContent = data.class_name;
                document.getElementById('confPill').textContent = (data.confidence * 100).toFixed(1) + '%';

                const grid = document.getElementById('scoresGrid');
                grid.innerHTML = '';
                data.all_scores.forEach((score, i) => {
                    const pct = (score * 100).toFixed(1);
                    const isTop = i === data.pred_idx;
                    grid.innerHTML += `
                        <div class="score-card ${isTop ? 'top-class' : ''}">
                            <div class="score-name">${data.class_names[i]}</div>
                            <div class="score-bar-bg">
                                <div class="score-bar-fill" style="width:${pct}%"></div>
                            </div>
                            <div class="score-pct">${pct}%</div>
                        </div>`;
                });

                document.getElementById('results').style.display = 'block';
                setTimeout(() => {
                    document.getElementById('results').scrollIntoView({behavior:'smooth', block:'center'});
                    document.getElementById('scrollUp').style.display = 'flex';
                }, 200);
            }
        } catch(e) {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('error').textContent = 'فشل الاتصال بالخادم';
            document.getElementById('error').style.display = 'block';
        }

        btn.disabled = false;
    }

    function clearAll() {
        selectedFile = null;
        document.getElementById('previewSection').style.display = 'none';
        document.getElementById('results').style.display = 'none';
        document.getElementById('error').style.display = 'none';
        document.getElementById('fileInput').value = '';
    }

    window.addEventListener('scroll', () => {
        document.getElementById('scrollUp').style.display = window.scrollY > 300 ? 'flex' : 'none';
    });

    // تحميل الصور عند بدء الصفحة
    fetch('/get_samples').then(r => r.json()).then(loadSampleImages);
</script>
</body>
</html>
'''

# ======================== Routes ========================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/get_samples')
def get_samples():
    samples = []
    for class_name in class_names:
        if class_name in sample_images:
            images = []
            for img_path in sample_images[class_name]:
                ext = img_path.suffix.lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                images.append({
                    'src': f"data:{mime};base64,{image_to_base64(img_path)}",
                    'path': str(img_path)
                })
            samples.append({'class_name': class_name, 'images': images})
    return jsonify({'samples': samples})


@app.route('/get_image', methods=['POST'])
def get_image():
    data = request.json
    path = data.get('path')
    return jsonify({'image': image_to_base64(path)})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ── تحميل الصورة ──
        if 'sample_path' in request.form:
            image_pil = Image.open(request.form['sample_path']).convert('RGB')
        else:
            image_pil = Image.open(BytesIO(request.files['image'].read())).convert('RGB')

        # ── معالجة الصورة ──
        image_resized = image_pil.resize((128, 128))
        img_array = np.array(image_resized) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) / \
                     torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        img_tensor = img_tensor.to(device)

        # ── Grad-CAM + Prediction ──
        cam, pred_idx, probabilities = gradcam.generate(img_tensor)

        # ── إنتاج صورة Overlay ──
        gradcam_b64 = apply_gradcam_overlay(image_resized, cam, alpha=0.45)

        return jsonify({
            'class_name': class_names[pred_idx],
            'confidence': float(probabilities[pred_idx]),
            'pred_idx': int(pred_idx),
            'all_scores': probabilities.tolist(),
            'class_names': class_names,
            'gradcam_image': gradcam_b64,   # ← الصورة المدمجة مع خريطة الحرارة
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)