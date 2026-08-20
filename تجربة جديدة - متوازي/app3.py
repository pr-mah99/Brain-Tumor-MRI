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
        spatial_weights = torch.sigmoid(self.spatial_conv(x))
        channel_pool = self.channel_avg(x)
        channel_weights = torch.sigmoid(self.channel_out(F.relu(self.channel_fc(channel_pool))))
        x_enhanced = x * spatial_weights * channel_weights
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
    """
    def __init__(self, channels):
        super().__init__()
        reduced_ch = max(4, channels // 4)
        self.context_net = nn.Sequential(
            nn.Conv2d(channels, reduced_ch, 1),
            nn.BatchNorm2d(reduced_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, reduced_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, channels, 3, padding=1),
            nn.Sigmoid()
        )
        self.lam = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        context = self.context_net(x)
        x_weighted = context * x
        y_max = F.max_pool2d(x_weighted, kernel_size=2, stride=2)
        y_avg = F.avg_pool2d(x_weighted, kernel_size=2, stride=2)
        lam_clamped = torch.sigmoid(self.lam)
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
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        probabilities = F.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam, class_idx, probabilities


def apply_gradcam_overlay(original_img_pil, cam, alpha=0.45):
    orig_np = np.array(original_img_pil)
    h, w = orig_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = (orig_np.astype(np.float32) * (1 - alpha) +
               heatmap_rgb.astype(np.float32) * alpha).astype(np.uint8)
    mask = (cam_resized > 0.70).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (255, 255, 255), 1)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
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


# ======================== NEW: Layer-by-layer activation visualizer ========================
# هذا الجزء الجديد يستخرج مخرجات كل طبقة أثناء التمرير الأمامي (forward pass)
# ويحوّلها إلى صور حرارية (heatmaps) لعرضها بشكل متحرك في الواجهة — مثالي لعرض المناقشة.

def feature_map_to_base64(fmap, size=180, apply_relu=True):
    """
    يحوّل خريطة ميزات [1, C, H, W] إلى صورة حرارية ملوّنة (JET) بحجم موحّد.
    نأخذ متوسط التنشيط عبر القنوات (channel-mean) كتمثيل مرئي للطبقة.
    """
    fm = fmap.detach()
    if apply_relu:
        fm = F.relu(fm)
    fm = fm[0]  # [C, H, W]
    mean_map = fm.mean(dim=0).cpu().numpy()  # [H, W]

    if mean_map.max() > mean_map.min():
        norm = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min())
    else:
        norm = np.zeros_like(mean_map)

    norm_resized = cv2.resize(norm, (size, size), interpolation=cv2.INTER_CUBIC)
    heat_uint8 = np.uint8(255 * norm_resized)
    heatmap_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    result_pil = Image.fromarray(heatmap_rgb)
    buf = BytesIO()
    result_pil.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode(), list(fmap.shape[1:])


# وصف كل طبقة باللغة العربية لعرضها أثناء المناقشة
LAYER_STAGES = [
    {"key": "conv1",       "title": "Conv1 + BatchNorm + ReLU", "voice": "الطبقة التلافيفية الأولى",
     "desc": "الطبقة التلافيفية الأولى: تستخرج ميزات أولية بسيطة مثل الحواف والتباينات في شدة البكسل."},
    {"key": "adaptive1",   "title": "MAFE-1 (Adaptive Feature Extractor)", "voice": "وحدة الانتباه التكيفي الأولى",
     "desc": "وحدة الانتباه التكيّفي المصغّرة: تدمج انتباهًا مكانيًا (Spatial) وقنواتيًا (Channel) بالتوازي لتعزيز المناطق المهمة."},
    {"key": "pool1",       "title": "CAIP-1 (Context-Aware Intelligent Pooling)", "voice": "طبقة التجميع الذكي الأولى",
     "desc": "تجميع ذكي واعٍ بالسياق: يُرجّح كل موقع مكانيًا قبل الدمج بين Max و Average Pooling عبر معامل λ قابل للتعلّم."},
    {"key": "conv2",       "title": "Conv2 + BatchNorm + ReLU", "voice": "الطبقة التلافيفية الثانية",
     "desc": "الطبقة التلافيفية الثانية: تبني تمثيلات أعمق اعتمادًا على الميزات المستخرجة من المرحلة السابقة."},
    {"key": "multiscale2", "title": "MMSP-2 (Multi-Scale Processor)", "voice": "وحدة المعالجة متعددة المقاييس الثانية",
     "desc": "معالجة متعددة المقاييس: تدمج مرشحات 1×1 و3×3 وconv عميق فصلي (Depthwise) لالتقاط أنماط بأحجام مختلفة."},
    {"key": "pool2",       "title": "CAIP-2", "voice": "طبقة التجميع الذكي الثانية",
     "desc": "تجميع ذكي ثانٍ يقلّص الأبعاد المكانية مع الحفاظ على المعلومات الأكثر أهمية للتصنيف."},
    {"key": "conv3",       "title": "Conv3 + BatchNorm + ReLU", "voice": "الطبقة التلافيفية الثالثة",
     "desc": "طبقة تلافيفية أعمق تستخرج ميزات دلالية أكثر تعقيدًا مرتبطة بشكل وملمس الورم."},
    {"key": "adaptive3",   "title": "MAFE-3", "voice": "وحدة الانتباه التكيفي الثانية",
     "desc": "وحدة انتباه تكيّفي ثانية تعمل على تمثيلات أعمق لتحديد أكثر المناطق تأثيرًا في القرار النهائي."},
    {"key": "pool3",       "title": "CAIP-3", "voice": "طبقة التجميع الذكي الثالثة",
     "desc": "المرحلة الأخيرة من التجميع الذكي قبل الطبقة التلافيفية العميقة النهائية."},
    {"key": "conv4",       "title": "Conv4 + BatchNorm + ReLU", "voice": "الطبقة التلافيفية العميقة الرابعة",
     "desc": "أعمق طبقة تلافيفية في الشبكة: تكوّن التمثيل النهائي عالي المستوى المستخدم في Grad-CAM."},
    {"key": "multiscale4", "title": "MMSP-4", "voice": "وحدة المعالجة متعددة المقاييس النهائية",
     "desc": "معالجة متعددة المقاييس نهائية تدمج جميع الأنماط المكتشفة قبل التصنيف."},
    {"key": "gradcam",     "title": "Grad-CAM Overlay", "voice": "خريطة غراد كام الحرارية",
     "desc": "خريطة حرارية توضح بدقة موقع الورم الذي اعتمد عليه النموذج في اتخاذ القرار النهائي."},
    {"key": "classifier",  "title": "Classifier (Fully Connected)", "voice": "طبقة التصنيف النهائية",
     "desc": "الطبقة النهائية المكتملة الاتصال: تحوّل الميزات المستخرجة إلى احتمالات الأصناف الأربعة."},
]


def capture_layer_activations(model, gradcam, img_tensor, original_resized_pil):
    """
    يسجّل hooks على كل طبقة رئيسية، يشغّل GradCAM.generate (الذي يقوم أصلاً
    بتمرير أمامي وخلفي)، ثم يبني قائمة صور توضيحية مرتبة حسب تسلسل الشبكة.
    """
    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            captured[name] = out.detach()
        return hook

    handles = []
    handles.append(model.bn1.register_forward_hook(make_hook("conv1")))
    handles.append(model.adaptive1.register_forward_hook(make_hook("adaptive1")))
    handles.append(model.pool1.register_forward_hook(make_hook("pool1")))
    handles.append(model.bn2.register_forward_hook(make_hook("conv2")))
    handles.append(model.multiscale2.register_forward_hook(make_hook("multiscale2")))
    handles.append(model.pool2.register_forward_hook(make_hook("pool2")))
    handles.append(model.bn3.register_forward_hook(make_hook("conv3")))
    handles.append(model.adaptive3.register_forward_hook(make_hook("adaptive3")))
    handles.append(model.pool3.register_forward_hook(make_hook("pool3")))
    handles.append(model.bn4.register_forward_hook(make_hook("conv4")))
    handles.append(model.multiscale4.register_forward_hook(make_hook("multiscale4")))

    try:
        cam, pred_idx, probabilities = gradcam.generate(img_tensor)
    finally:
        for h in handles:
            h.remove()

    gradcam_b64 = apply_gradcam_overlay(original_resized_pil, cam, alpha=0.45)

    layers_output = []
    for stage in LAYER_STAGES:
        key = stage["key"]
        if key == "gradcam":
            layers_output.append({
                "key": key, "title": stage["title"], "desc": stage["desc"],
                "voice": stage["voice"], "image": gradcam_b64, "shape": None
            })
        elif key == "classifier":
            continue  # النتيجة النهائية تُعرض في قسم النتائج المنفصل أصلاً
        elif key in captured:
            apply_relu = key not in ("pool1", "pool2", "pool3")  # التجميع مخرَج موجب أصلاً
            img_b64, shape = feature_map_to_base64(captured[key], apply_relu=apply_relu)
            layers_output.append({
                "key": key, "title": stage["title"], "desc": stage["desc"],
                "voice": stage["voice"], "image": img_b64, "shape": shape
            })

    return layers_output, cam, pred_idx, probabilities


# ======================== Flask App ========================
app = Flask(__name__)
device = torch.device('cpu')

model = UltraLightPatentBrainTumorCNN(num_classes=4).to(device)
checkpoint = torch.load('best_brain_tumor_model_patent.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])

gradcam = GradCAM(model, target_layer=model.conv4)


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
    <title>تصنيف أورام الدماغ + Grad-CAM + عرض الطبقات</title>
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
            max-width: 980px;
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

        .header h1 { font-size: 30px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 6px; }
        .header p { font-size: 13px; opacity: 0.85; letter-spacing: 0.5px; }

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

        .tabs { display: flex; gap: 8px; margin-bottom: 28px; }

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

        .sample-item { cursor: pointer; border-radius: 12px; overflow: hidden; border: 2px solid transparent; transition: all 0.3s; }
        .sample-item:hover { border-color: #667eea; transform: translateY(-5px); box-shadow: 0 10px 24px rgba(102,126,234,0.4); }
        .sample-item img { width: 100%; height: 110px; object-fit: cover; display: block; }

        .preview-section { display: none; margin: 24px 0 0; }

        .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }

        .img-card { border-radius: 16px; overflow: hidden; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.08); }

        .img-card-label {
            padding: 10px 14px;
            font-size: 12px;
            font-weight: 700;
            color: rgba(255,255,255,0.6);
            letter-spacing: 0.8px;
            text-transform: uppercase;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            display: flex; align-items: center; gap: 8px;
        }

        .label-dot { width: 8px; height: 8px; border-radius: 50%; }
        .dot-original { background: #60a5fa; }
        .dot-gradcam  { background: #f97316; }

        .img-card img { width: 100%; height: 340px; object-fit: contain; display: block; background: #0a0a0a; }

        .img-placeholder {
            width: 100%; height: 340px; background: rgba(255,255,255,0.03);
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            color: rgba(255,255,255,0.2); font-size: 32px; gap: 8px;
        }
        .img-placeholder span { font-size: 11px; letter-spacing: 0.5px; }

        .controls { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }

        button {
            flex: 1; padding: 14px; border: none; border-radius: 12px;
            font-weight: 700; cursor: pointer; font-size: 15px; transition: all 0.25s;
        }

        .btn-classify {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; box-shadow: 0 6px 20px rgba(102,126,234,0.35);
        }
        .btn-classify:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(102,126,234,0.5); }
        .btn-classify:disabled { opacity: 0.6; cursor: not-allowed; }

        .btn-clear {
            flex: 0.4; background: rgba(255,255,255,0.07); color: rgba(255,255,255,0.6);
            border: 1px solid rgba(255,255,255,0.12);
        }
        .btn-clear:hover { background: rgba(255,255,255,0.12); color: white; }

        /* ── إعدادات الصوت (جديد) ── */
        .voice-settings {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .voice-option {
            flex: 1;
            min-width: 220px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 12px 16px;
        }

        .voice-option-label {
            font-size: 13px;
            font-weight: 600;
            color: rgba(255,255,255,0.7);
        }

        .voice-option-sub {
            font-size: 11px;
            color: rgba(255,255,255,0.35);
            margin-top: 2px;
        }

        .switch {
            position: relative;
            display: inline-block;
            width: 46px;
            height: 26px;
            flex-shrink: 0;
        }
        .switch input { display: none; }
        .switch-slider {
            position: absolute;
            cursor: pointer;
            inset: 0;
            background: rgba(255,255,255,0.15);
            border-radius: 26px;
            transition: 0.3s;
        }
        .switch-slider::before {
            content: '';
            position: absolute;
            height: 20px; width: 20px;
            left: 3px; bottom: 3px;
            background: white;
            border-radius: 50%;
            transition: 0.3s;
        }
        .switch input:checked + .switch-slider {
            background: linear-gradient(135deg, #667eea, #764ba2);
        }
        .switch input:checked + .switch-slider::before {
            transform: translateX(20px);
        }
        .switch input:disabled + .switch-slider {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .loading { display: none; text-align: center; padding: 32px; }

        .scanner-ring {
            width: 60px; height: 60px; border: 3px solid rgba(102,126,234,0.2);
            border-top-color: #667eea; border-radius: 50%;
            animation: spin 0.8s linear infinite; margin: 0 auto 14px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text { color: rgba(255,255,255,0.55); font-size: 14px; animation: fade 1.2s ease-in-out infinite; }
        @keyframes fade { 0%,100% { opacity:0.4; } 50% { opacity:1; } }

        /* ── Pipeline Animation (جديد) ── */
        .pipeline-section {
            display: none;
            margin-top: 12px;
            position: relative;
            border-radius: 22px;
            padding: 22px;
            background:
                radial-gradient(circle at 15% 20%, rgba(102,126,234,0.10), transparent 45%),
                radial-gradient(circle at 85% 80%, rgba(249,115,22,0.08), transparent 45%),
                rgba(255,255,255,0.015);
            border: 1px solid rgba(102,126,234,0.15);
            overflow: hidden;
        }

        .pipeline-section::before {
            content: '';
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(102,126,234,0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(102,126,234,0.05) 1px, transparent 1px);
            background-size: 26px 26px;
            animation: gridDrift 14s linear infinite;
            pointer-events: none;
            opacity: 0.5;
        }

        @keyframes gridDrift {
            from { background-position: 0 0; }
            to { background-position: 260px 260px; }
        }

        .pipeline-progress-bar {
            height: 8px;
            background: rgba(255,255,255,0.08);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 18px;
            position: relative;
            box-shadow: inset 0 0 6px rgba(0,0,0,0.4);
        }
        .pipeline-progress-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #667eea, #a855f7, #f97316);
            background-size: 200% 100%;
            animation: progressShimmer 1.4s linear infinite;
            transition: width 0.5s cubic-bezier(0.22,1,0.36,1);
            box-shadow: 0 0 14px rgba(102,126,234,0.7);
            position: relative;
        }
        @keyframes progressShimmer {
            0% { background-position: 0% 0; }
            100% { background-position: -200% 0; }
        }

        .pipeline-track {
            display: flex;
            align-items: center;
            gap: 4px;
            overflow-x: auto;
            padding-bottom: 14px;
            margin-bottom: 22px;
            position: relative;
            z-index: 1;
        }

        .pipeline-chip {
            flex: 0 0 auto;
            padding: 8px 14px;
            border-radius: 20px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            font-size: 11px;
            font-weight: 700;
            color: rgba(255,255,255,0.35);
            white-space: nowrap;
            transition: all 0.4s cubic-bezier(0.34,1.56,0.64,1);
            position: relative;
        }
        .pipeline-chip.done {
            background: rgba(102,126,234,0.18);
            border-color: rgba(102,126,234,0.4);
            color: #a5b4fc;
        }
        .pipeline-chip.done::after {
            content: '✓';
            margin-right: 5px;
            color: #34d399;
        }
        .pipeline-chip.current {
            background: linear-gradient(135deg, #667eea, #764ba2, #f97316);
            background-size: 200% 200%;
            animation: chipPulse 0.9s ease-in-out infinite, chipGradient 1.6s ease infinite;
            border-color: transparent;
            color: white;
            transform: scale(1.15);
        }
        @keyframes chipPulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(102,126,234,0.6), 0 6px 18px rgba(102,126,234,0.5); }
            50% { box-shadow: 0 0 0 8px rgba(102,126,234,0), 0 6px 24px rgba(249,115,22,0.5); }
        }
        @keyframes chipGradient {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        .pipeline-connector {
            flex: 0 0 auto;
            width: 20px;
            height: 2px;
            background: rgba(255,255,255,0.1);
            position: relative;
            overflow: visible;
        }
        .pipeline-connector.active::before {
            content: '';
            position: absolute;
            top: -2px;
            left: 0;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #f97316;
            box-shadow: 0 0 8px 2px rgba(249,115,22,0.9);
            animation: flowDot 0.7s linear infinite;
        }
        @keyframes flowDot {
            0% { left: 0; opacity: 0; }
            15% { opacity: 1; }
            85% { opacity: 1; }
            100% { left: 18px; opacity: 0; }
        }

        .pipeline-stage-card {
            display: grid;
            grid-template-columns: 200px 1fr;
            gap: 20px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(102,126,234,0.3);
            border-radius: 18px;
            padding: 20px;
            align-items: center;
            opacity: 0;
            position: relative;
            z-index: 1;
            transform: translateY(18px) scale(0.95) rotateX(4deg);
            animation: stageIn 0.6s cubic-bezier(0.22,1,0.36,1) forwards, cardGlow 2s ease-in-out infinite;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }

        @keyframes stageIn {
            to { opacity: 1; transform: translateY(0) scale(1) rotateX(0); }
        }

        @keyframes cardGlow {
            0%, 100% { border-color: rgba(102,126,234,0.3); }
            50% { border-color: rgba(249,115,22,0.45); }
        }

        .pipeline-stage-img-wrap {
            border-radius: 14px;
            overflow: hidden;
            background: #0a0a0a;
            border: 1px solid rgba(255,255,255,0.08);
            position: relative;
        }

        .pipeline-stage-img-wrap img {
            width: 100%;
            height: 160px;
            object-fit: cover;
            display: block;
            animation: imgReveal 0.8s ease;
        }

        @keyframes imgReveal {
            from { filter: brightness(2) blur(6px); transform: scale(1.15); }
            to { filter: brightness(1) blur(0); transform: scale(1); }
        }

        .scan-line {
            position: absolute;
            left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, transparent, #34d399, transparent);
            box-shadow: 0 0 10px 2px rgba(52,211,153,0.8);
            animation: scanSweep 1s ease-in-out infinite;
            pointer-events: none;
        }
        @keyframes scanSweep {
            0% { top: 0; opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }

        .pipeline-stage-title {
            font-size: 17px;
            font-weight: 800;
            color: #a5b4fc;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .pipeline-stage-title .stage-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea, #f97316);
            color: white;
            font-size: 12px;
            flex-shrink: 0;
            animation: numPop 0.5s cubic-bezier(0.34,1.56,0.64,1);
        }
        @keyframes numPop {
            0% { transform: scale(0) rotate(-180deg); }
            100% { transform: scale(1) rotate(0); }
        }

        .pipeline-stage-desc {
            font-size: 13.5px;
            line-height: 1.9;
            color: rgba(255,255,255,0.65);
            border-right: 2px solid rgba(102,126,234,0.5);
            padding-right: 10px;
            min-height: 1.9em;
        }

        .typing-cursor {
            display: inline-block;
            width: 2px;
            height: 14px;
            background: #a5b4fc;
            margin-right: 2px;
            animation: blink 0.8s step-start infinite;
            vertical-align: middle;
        }
        @keyframes blink { 50% { opacity: 0; } }

        .pipeline-stage-shape {
            display: inline-block;
            margin-top: 10px;
            font-size: 11px;
            font-weight: 700;
            color: rgba(255,255,255,0.4);
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 3px 10px;
            direction: ltr;
            opacity: 0;
            animation: fadeInDelay 0.4s ease 0.5s forwards;
        }
        @keyframes fadeInDelay { to { opacity: 1; } }

        .pipeline-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            gap: 10px;
            margin-top: 18px;
            position: relative;
            z-index: 1;
        }

        .pipeline-gallery-item {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(0,0,0,0.25);
            opacity: 0;
            animation: galleryIn 0.6s cubic-bezier(0.34,1.56,0.64,1) forwards;
            transition: transform 0.25s ease, border-color 0.25s ease;
        }
        .pipeline-gallery-item:hover {
            transform: translateY(-4px) scale(1.04);
            border-color: rgba(102,126,234,0.5);
        }
        @keyframes galleryIn {
            0% { opacity: 0; transform: translateY(20px) scale(0.8) rotate(-3deg); }
            100% { opacity: 1; transform: translateY(0) scale(1) rotate(0); }
        }
        .pipeline-gallery-item img { width: 100%; height: 90px; object-fit: cover; display: block; }
        .pipeline-gallery-item .g-label {
            font-size: 10px;
            font-weight: 700;
            color: rgba(255,255,255,0.55);
            padding: 6px 8px;
            text-align: center;
        }

        /* ── Confetti (احتفال بالنتيجة النهائية) ── */
        #confettiCanvas {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            pointer-events: none;
            z-index: 9999;
        }

        .result-header.pulse-once {
            animation: resultPop 0.6s cubic-bezier(0.34,1.56,0.64,1);
        }
        @keyframes resultPop {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(102,126,234,0.6); }
            50% { transform: scale(1.03); }
            100% { transform: scale(1); }
        }

        @media (max-width: 600px) {
            .pipeline-stage-card { grid-template-columns: 1fr; }
        }

        /* ── تصميم مكبّر خاص لمرحلة Grad-CAM ── */
        .pipeline-stage-card.gradcam-stage {
            grid-template-columns: 1fr;
            border-color: rgba(249,115,22,0.5);
            box-shadow: 0 0 30px rgba(249,115,22,0.25), 0 10px 30px rgba(0,0,0,0.35);
        }
        .pipeline-stage-card.gradcam-stage .pipeline-stage-img-wrap img {
            height: 380px;
            object-fit: contain;
            background: #0a0a0a;
        }
        .pipeline-stage-card.gradcam-stage .pipeline-stage-title {
            font-size: 20px;
            color: #fdba74;
        }
        .pipeline-stage-card.gradcam-stage .pipeline-stage-title .stage-num {
            background: linear-gradient(135deg, #f97316, #ef4444);
        }

        /* ── Results ── */
        .results { display: none; margin-top: 8px; }

        .result-header {
            display: flex; align-items: center; justify-content: space-between;
            background: linear-gradient(135deg, rgba(102,126,234,0.18), rgba(118,75,162,0.18));
            border: 1px solid rgba(102,126,234,0.35);
            border-radius: 16px; padding: 20px 24px; margin-bottom: 16px;
        }

        .result-label { font-size: 11px; font-weight: 700; color: rgba(255,255,255,0.45); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
        .result-value { font-size: 30px; font-weight: 800; color: #a5b4fc; }

        .confidence-pill {
            background: rgba(102,126,234,0.25); border: 1px solid rgba(102,126,234,0.45);
            border-radius: 20px; padding: 8px 20px; font-size: 22px; font-weight: 800; color: white;
        }

        .scores-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 4px; }

        .score-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 12px 16px; transition: all 0.2s; }
        .score-card.top-class { background: rgba(102,126,234,0.15); border-color: rgba(102,126,234,0.4); }
        .score-name { font-size: 12px; color: rgba(255,255,255,0.5); margin-bottom: 8px; font-weight: 600; }
        .score-bar-bg { height: 6px; background: rgba(255,255,255,0.08); border-radius: 6px; overflow: hidden; margin-bottom: 6px; }
        .score-bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.6s cubic-bezier(0.22,1,0.36,1); }
        .score-pct { font-size: 18px; font-weight: 800; color: white; }

        .gradcam-legend {
            display: flex; align-items: center; gap: 8px; padding: 10px 14px;
            background: rgba(0,0,0,0.25); border-top: 1px solid rgba(255,255,255,0.06);
            font-size: 11px; color: rgba(255,255,255,0.45); flex-wrap: wrap;
        }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .legend-swatch { width: 22px; height: 8px; border-radius: 4px; }

        .error {
            display: none; background: rgba(220,38,38,0.15); color: #fca5a5; padding: 14px 18px;
            border-radius: 12px; border-left: 4px solid #ef4444; margin-top: 16px;
        }

        .section-title {
            color: rgba(255,255,255,0.55); font-size: 12px; font-weight: 700;
            letter-spacing: 1px; text-transform: uppercase; margin-bottom: 14px;
        }

        .sample-class-title {
            color: #a5b4fc; font-size: 14px; font-weight: 700; margin-bottom: 10px;
            padding-bottom: 6px; border-bottom: 1px solid rgba(102,126,234,0.2);
        }

        .scroll-up {
            position: fixed; bottom: 28px; right: 28px; width: 48px; height: 48px;
            background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none;
            border-radius: 50%; cursor: pointer; font-size: 20px; display: none;
            align-items: center; justify-content: center; box-shadow: 0 6px 20px rgba(102,126,234,0.5);
            transition: all 0.3s; z-index: 999; flex: 0; padding: 0;
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
        <div class="badge">✦ عرض حيّ لكل طبقات الشبكة (Layer-by-Layer Animation)</div>
    </div>

    <div class="content">
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('upload', this)">📤 رفع صورة</button>
            <button class="tab-btn" onclick="switchTab('samples', this)">🗂 صور عينات</button>
        </div>

        <div class="tab-content active" id="tab-upload">
            <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*">
                <div class="upload-icon">📤</div>
                <div class="upload-text">اضغط لرفع صورة MRI</div>
                <div class="upload-subtext">PNG · JPG · WEBP مقبول</div>
            </div>
        </div>

        <div class="tab-content" id="tab-samples">
            <div id="samplesContainer"></div>
        </div>

        <!-- ── إعدادات الصوت (جديد): تفعيل/تعطيل القراءة الصوتية + وضع "النتيجة فقط" ── -->
        <div class="voice-settings">
            <div class="voice-option">
                <div>
                    <div class="voice-option-label">🔊 تفعيل القراءة الصوتية</div>
                    <div class="voice-option-sub">تشغيل/إيقاف كل النطق في التطبيق</div>
                </div>
                <label class="switch">
                    <input type="checkbox" id="voiceEnabledToggle" checked onchange="onVoiceEnabledChange()">
                    <span class="switch-slider"></span>
                </label>
            </div>
            <div class="voice-option">
                <div>
                    <div class="voice-option-label">🎯 نطق النتيجة النهائية فقط</div>
                    <div class="voice-option-sub">بدون شرح كل طبقة أثناء المعالجة</div>
                </div>
                <label class="switch">
                    <input type="checkbox" id="resultOnlyToggle" checked onchange="onResultOnlyChange()">
                    <span class="switch-slider"></span>
                </label>
            </div>
        </div>

        <div class="preview-section" id="previewSection">
            <div class="comparison-grid">
                <div class="img-card">
                    <div class="img-card-label"><span class="label-dot dot-original"></span>الصورة الأصلية</div>
                    <img id="originalImg" src="" alt="Original">
                </div>
                <div class="img-card">
                    <div class="img-card-label"><span class="label-dot dot-gradcam"></span>Grad-CAM — منطقة الورم</div>
                    <div class="img-placeholder" id="gradcamPlaceholder">🔬 <span>يظهر بعد التصنيف</span></div>
                    <img id="gradcamImg" src="" alt="Grad-CAM" style="display:none">
                </div>
            </div>

            <div class="gradcam-legend">
                <span>شدة التنشيط:</span>
                <div class="legend-item"><div class="legend-swatch" style="background:linear-gradient(90deg,#00008b,#0000ff)"></div><span>منخفض</span></div>
                <div class="legend-item"><div class="legend-swatch" style="background:linear-gradient(90deg,#008000,#ffff00)"></div><span>متوسط</span></div>
                <div class="legend-item"><div class="legend-swatch" style="background:linear-gradient(90deg,#ffa500,#ff0000)"></div><span>عالي (منطقة الورم)</span></div>
                <span style="margin-right:auto; color:rgba(255,255,255,0.3)">الحدود البيضاء = تنشيط > 70%</span>
            </div>

            <div class="controls" style="margin-top:16px">
                <button class="btn-classify" id="classifyBtn" onclick="classify()">🔍 تصنيف + عرض الطبقات</button>
                <button class="btn-clear" onclick="clearAll()">✕ مسح</button>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="scanner-ring"></div>
            <div class="loading-text">جاري التحليل عبر جميع طبقات الشبكة...</div>
        </div>

        <!-- ── قسم عرض الطبقات المتحرك (جديد) ── -->
        <div class="pipeline-section" id="pipelineSection">
            <div class="section-title">🎬 رحلة الصورة عبر طبقات الشبكة</div>
            <div class="pipeline-progress-bar"><div class="pipeline-progress-fill" id="pipelineProgress"></div></div>
            <div class="pipeline-track" id="pipelineTrack"></div>
            <div id="pipelineStageArea"></div>
            <div class="pipeline-gallery" id="pipelineGallery"></div>
        </div>

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
<canvas id="confettiCanvas"></canvas>

<script>
    let selectedFile = null;
    let voiceEnabled = true;      // تفعيل/تعطيل القراءة الصوتية بالكامل
    let resultOnlyMode = true;   // نطق النتيجة النهائية فقط بدون شرح الطبقات
    let arabicVoice = null;

    // ── تحميل الأصوات العربية المتاحة في المتصفح ──
    function loadArabicVoice() {
        const voices = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
        arabicVoice = voices.find(v => v.lang && v.lang.toLowerCase().startsWith('ar')) || null;
    }
    if ('speechSynthesis' in window) {
        loadArabicVoice();
        window.speechSynthesis.onvoiceschanged = loadArabicVoice;
    }

    // ── تفعيل/تعطيل الصوت بالكامل ──
    function onVoiceEnabledChange() {
        voiceEnabled = document.getElementById('voiceEnabledToggle').checked;
        if (!voiceEnabled && 'speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        // عند تعطيل الصوت كليًا، عطّل خيار "النتيجة فقط" لأنه لا معنى له بدون صوت أصلاً
        document.getElementById('resultOnlyToggle').disabled = !voiceEnabled;
    }

    // ── نطق النتيجة النهائية فقط بدون شرح كل طبقة ──
    function onResultOnlyChange() {
        resultOnlyMode = document.getElementById('resultOnlyToggle').checked;
    }

    // ── نطق نص عربي، يُرجع Promise تنتهي عند اكتمال النطق ──
    function speakText(text) {
        return new Promise(resolve => {
            if (!voiceEnabled || !('speechSynthesis' in window) || !text) {
                resolve();
                return;
            }
            try {
                const utter = new SpeechSynthesisUtterance(text);
                utter.lang = 'ar-SA';
                if (arabicVoice) utter.voice = arabicVoice;
                utter.rate = 1.02;
                utter.pitch = 1.0;
                utter.onend = () => resolve();
                utter.onerror = () => resolve();
                window.speechSynthesis.speak(utter);
            } catch (e) {
                resolve();
            }
        });
    }

    // ترجمة أسماء الأصناف الإنجليزية إلى نطق عربي واضح
    const CLASS_NAME_AR = {
        'glioma': 'ورم دبقي',
        'meningioma': 'ورم سحائي',
        'notumor': 'لا يوجد ورم',
        'pituitary': 'ورم نخامي'
    };

    function switchTab(name, btn) {
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('tab-' + name).classList.add('active');
        btn.classList.add('active');
    }

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
        document.getElementById('pipelineSection').style.display = 'none';
        switchTab('upload', document.querySelector('.tab-btn'));
        setTimeout(() => document.getElementById('previewSection').scrollIntoView({behavior:'smooth'}), 100);
    }

    function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

    // ── تأثير الكتابة الحية للوصف ──
    async function typeText(el, text, totalMs) {
        el.innerHTML = '';
        const cursor = document.createElement('span');
        cursor.className = 'typing-cursor';
        const perChar = Math.max(6, Math.floor(totalMs / Math.max(text.length, 1)));
        let buffer = '';
        for (let i = 0; i < text.length; i++) {
            buffer += text[i];
            el.textContent = buffer;
            el.appendChild(cursor);
            await sleep(perChar);
        }
        cursor.remove();
    }

    // ── تشغيل رسوم متحركة لكل طبقة بفاصل ~1 ثانية ──
    // ملاحظة: الأنميشن المرئي (الصور، الكتابة، التقدم) يعمل دائمًا؛
    // فقط النطق الصوتي لكل طبقة يُتخطّى إذا كان وضع "النتيجة فقط" مفعّلاً.
    async function playPipelineAnimation(layers) {
        const track = document.getElementById('pipelineTrack');
        const stageArea = document.getElementById('pipelineStageArea');
        const gallery = document.getElementById('pipelineGallery');
        const progress = document.getElementById('pipelineProgress');

        track.innerHTML = '';
        stageArea.innerHTML = '';
        gallery.innerHTML = '';
        progress.style.width = '0%';

        layers.forEach((layer, i) => {
            const chip = document.createElement('div');
            chip.className = 'pipeline-chip';
            chip.id = 'chip-' + i;
            chip.textContent = layer.title.split('(')[0].trim();
            track.appendChild(chip);

            if (i < layers.length - 1) {
                const connector = document.createElement('div');
                connector.className = 'pipeline-connector';
                connector.id = 'conn-' + i;
                track.appendChild(connector);
            }
        });

        document.getElementById('pipelineSection').style.display = 'block';
        document.getElementById('pipelineSection').scrollIntoView({behavior:'smooth', block:'start'});

        // في وضع "النتيجة فقط" نسرّع المرور بين الطبقات بما أنه لا يوجد نطق يحدد الإيقاع
        const stepDelayMs = resultOnlyMode ? 450 : 1000;

        for (let i = 0; i < layers.length; i++) {
            const layer = layers[i];

            document.querySelectorAll('.pipeline-chip').forEach(c => c.classList.remove('current'));
            document.querySelectorAll('.pipeline-connector').forEach(c => c.classList.remove('active'));
            const chip = document.getElementById('chip-' + i);
            chip.classList.add('current');
            const prevConn = document.getElementById('conn-' + (i - 1));
            if (prevConn) prevConn.classList.add('active');

            const shapeText = layer.shape ? `Shape: [${layer.shape.join(' × ')}]` : '';
            const isGradcam = layer.key === 'gradcam';

            stageArea.innerHTML = `
                <div class="pipeline-stage-card ${isGradcam ? 'gradcam-stage' : ''}">
                    <div class="pipeline-stage-img-wrap">
                        <img src="data:image/jpeg;base64,${layer.image}" alt="${layer.title}">
                        <div class="scan-line"></div>
                    </div>
                    <div>
                        <div class="pipeline-stage-title"><span class="stage-num">${i+1}</span> ${layer.title}</div>
                        <div class="pipeline-stage-desc" id="stageDesc"></div>
                        ${shapeText ? `<div class="pipeline-stage-shape">${shapeText}</div>` : ''}
                    </div>
                </div>`;

            const descEl = document.getElementById('stageDesc');
            typeText(descEl, layer.desc, 900);

            const galItem = document.createElement('div');
            galItem.className = 'pipeline-gallery-item';
            galItem.style.animationDelay = '0.03s';
            galItem.innerHTML = `
                <img src="data:image/jpeg;base64,${layer.image}" alt="${layer.title}">
                <div class="g-label">${i+1}. ${layer.title.split('(')[0].trim()}</div>`;
            gallery.appendChild(galItem);

            progress.style.width = `${Math.round(((i+1)/layers.length)*100)}%`;

            // النطق الصوتي لاسم الطبقة العربي المختصر — يُتخطّى في وضع "النتيجة فقط"
            if (resultOnlyMode) {
                await sleep(stepDelayMs);
            } else {
                await Promise.all([sleep(stepDelayMs), speakText(layer.voice || layer.title)]);
            }

            chip.classList.remove('current');
            chip.classList.add('done');
        }

        document.querySelectorAll('.pipeline-connector').forEach(c => c.classList.remove('active'));
    }

    // ── قصاصات احتفالية عند ظهور النتيجة ──
    function fireConfetti() {
        const canvas = document.getElementById('confettiCanvas');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        const ctx = canvas.getContext('2d');
        const colors = ['#667eea', '#764ba2', '#f97316', '#34d399', '#a5b4fc'];
        const pieces = Array.from({length: 90}, () => ({
            x: Math.random() * canvas.width,
            y: -20 - Math.random() * canvas.height * 0.3,
            r: 4 + Math.random() * 5,
            c: colors[Math.floor(Math.random() * colors.length)],
            vy: 2 + Math.random() * 3,
            vx: -1.5 + Math.random() * 3,
            rot: Math.random() * 360,
            vrot: -6 + Math.random() * 12
        }));

        let frame = 0;
        const maxFrames = 130;
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            pieces.forEach(p => {
                p.x += p.vx; p.y += p.vy; p.rot += p.vrot;
                ctx.save();
                ctx.translate(p.x, p.y);
                ctx.rotate(p.rot * Math.PI / 180);
                ctx.fillStyle = p.c;
                ctx.fillRect(-p.r/2, -p.r/2, p.r, p.r * 0.6);
                ctx.restore();
            });
            frame++;
            if (frame < maxFrames) {
                requestAnimationFrame(draw);
            } else {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        }
        draw();
    }

    async function classify() {
        if (!selectedFile) return;
        const btn = document.getElementById('classifyBtn');
        btn.disabled = true;

        document.getElementById('results').style.display = 'none';
        document.getElementById('error').style.display = 'none';
        document.getElementById('pipelineSection').style.display = 'none';
        document.getElementById('loading').style.display = 'block';

        document.getElementById('gradcamImg').style.display = 'none';
        document.getElementById('gradcamPlaceholder').style.display = 'flex';

        // ── نطق بداية التشخيص — يُتخطّى في وضع "النتيجة فقط" ──
        const introSpeechPromise = resultOnlyMode
            ? Promise.resolve()
            : speakText('بدء تشخيص الورم، جاري تحليل الصورة عبر جميع طبقات الشبكة العصبية.');

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
                // التأكد من انتهاء نطق المقدمة قبل بدء نطق الطبقات لتفادي التداخل
                await introSpeechPromise;

                // تشغيل الأنميشن عبر كل الطبقات أولاً (الصور والنصوص تظهر دائمًا،
                // والنطق الصوتي لكل طبقة يعتمد على وضع "النتيجة فقط")
                await playPipelineAnimation(data.layers);

                // إظهار Grad-CAM في بطاقة المقارنة العلوية أيضًا
                document.getElementById('gradcamImg').src = 'data:image/jpeg;base64,' + data.gradcam_image;
                document.getElementById('gradcamImg').style.display = 'block';
                document.getElementById('gradcamPlaceholder').style.display = 'none';

                document.getElementById('resultValue').textContent =
                    data.class_name + ' — ' + (CLASS_NAME_AR[data.class_name] || '');
                document.getElementById('confPill').textContent = (data.confidence * 100).toFixed(1) + '%';

                const grid = document.getElementById('scoresGrid');
                grid.innerHTML = '';
                data.all_scores.forEach((score, i) => {
                    const pct = (score * 100).toFixed(1);
                    const isTop = i === data.pred_idx;
                    grid.innerHTML += `
                        <div class="score-card ${isTop ? 'top-class' : ''}">
                            <div class="score-name">${data.class_names[i]}</div>
                            <div class="score-bar-bg"><div class="score-bar-fill" style="width:${pct}%"></div></div>
                            <div class="score-pct">${pct}%</div>
                        </div>`;
                });

                document.getElementById('results').style.display = 'block';
                const resultHeader = document.querySelector('.result-header');
                resultHeader.classList.remove('pulse-once');
                void resultHeader.offsetWidth; // إعادة تشغيل الأنميشن
                resultHeader.classList.add('pulse-once');
                fireConfetti();

                // ── نطق النتيجة النهائية: يُنطق دائمًا طالما الصوت مفعّل بشكل عام ──
                const arName = CLASS_NAME_AR[data.class_name] || data.class_name;
                const confPct = (data.confidence * 100).toFixed(1);
                const resultSentence = (data.class_name === 'notumor')
                    ? `اكتمل التحليل. النتيجة: No Tumor، لا يوجد ورم، بنسبة ثقة ${confPct} بالمئة.`
                    : `اكتمل التحليل. التشخيص هو: ${data.class_name}، بالعربية: ${arName}، بنسبة ثقة ${confPct} بالمئة.`;
                speakText(resultSentence);

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
        if ('speechSynthesis' in window) window.speechSynthesis.cancel();
        document.getElementById('previewSection').style.display = 'none';
        document.getElementById('results').style.display = 'none';
        document.getElementById('pipelineSection').style.display = 'none';
        document.getElementById('error').style.display = 'none';
        document.getElementById('fileInput').value = '';
    }

    window.addEventListener('scroll', () => {
        document.getElementById('scrollUp').style.display = window.scrollY > 300 ? 'flex' : 'none';
    });

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
        if 'sample_path' in request.form:
            image_pil = Image.open(request.form['sample_path']).convert('RGB')
        else:
            image_pil = Image.open(BytesIO(request.files['image'].read())).convert('RGB')

        image_resized = image_pil.resize((128, 128))
        img_array = np.array(image_resized) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) / \
                     torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        img_tensor = img_tensor.to(device)

        # ── التقاط جميع الطبقات + Grad-CAM + التنبؤ في تمرير واحد ──
        layers_output, cam, pred_idx, probabilities = capture_layer_activations(
            model, gradcam, img_tensor, image_resized
        )

        gradcam_b64 = apply_gradcam_overlay(image_resized, cam, alpha=0.45)

        return jsonify({
            'class_name': class_names[pred_idx],
            'confidence': float(probabilities[pred_idx]),
            'pred_idx': int(pred_idx),
            'all_scores': probabilities.tolist(),
            'class_names': class_names,
            'gradcam_image': gradcam_b64,
            'layers': layers_output,   # ← قائمة الطبقات لعرضها بالأنميشن
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)