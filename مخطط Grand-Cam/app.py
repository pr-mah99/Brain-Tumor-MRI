"""
Grad-CAM visualization for the 4 brain-tumor classes
(glioma, meningioma, notumor, pituitary).

For each class it picks one sample image, runs the model + Grad-CAM,
and plots:
    Col 1: original MRI
    Col 2: raw Grad-CAM heatmap
    Col 3: overlay + high-activation contour
    Col 4: bar chart of class probabilities

Reuses the model architecture & GradCAM class from your app.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button
import matplotlib as mpl

# ======================== Modern light theme ========================
BG_COLOR      = '#f7f8fc'
PANEL_COLOR   = '#ffffff'
ACCENT        = '#6d5ce7'      # violet
ACCENT2       = '#fb923c'      # orange
TEXT_MAIN     = '#1f2333'
TEXT_MUTED    = '#6b7280'
GRID_COLOR    = '#e5e7eb'
BAR_TOP       = '#6d5ce7'
BAR_OTHER     = '#e2e4f0'

mpl.rcParams['figure.facecolor'] = BG_COLOR
mpl.rcParams['axes.facecolor']   = PANEL_COLOR
mpl.rcParams['savefig.facecolor'] = BG_COLOR
mpl.rcParams['text.color']       = TEXT_MAIN
mpl.rcParams['axes.edgecolor']   = GRID_COLOR
mpl.rcParams['axes.labelcolor']  = TEXT_MUTED
mpl.rcParams['xtick.color']      = TEXT_MUTED
mpl.rcParams['ytick.color']      = TEXT_MUTED
mpl.rcParams['font.family']      = 'DejaVu Sans'

# ======================== Model (same as app) ========================
class MicroAdaptiveFeatureExtractor(nn.Module):
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
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam, class_idx, probs


# ======================== Helpers ========================
def preprocess_image(image_pil, size=128):
    image_resized = image_pil.resize((size, size))
    arr = np.array(image_resized) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / \
             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return tensor, image_resized


def make_overlay_with_contour(original_pil, cam, alpha=0.45, thresh=0.70):
    orig = np.array(original_pil)
    h, w = orig.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (orig.astype(np.float32) * (1 - alpha) +
               heatmap_rgb.astype(np.float32) * alpha).astype(np.uint8)

    mask = (cam_resized > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (255, 255, 255), 1)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay, heatmap_rgb


# ======================== Main ========================
def main():
    # ---- CONFIG: edit these paths ----
    MODEL_PATH = 'best_brain_tumor_model_patent.pth'
    SAMPLE_DIR = Path('sample_images')   # expects subfolders: glioma, meningioma, notumor, pituitary
    OUTPUT_PATH = 'gradcam_4classes.png'
    # -----------------------------------

    device = torch.device('cpu')

    model = UltraLightPatentBrainTumorCNN(num_classes=4).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])
    gradcam = GradCAM(model, target_layer=model.conv4)

    # pick one sample image per class
    sample_paths = {}
    for cname in class_names:
        cdir = SAMPLE_DIR / cname
        imgs = list(cdir.glob('*.jpg')) + list(cdir.glob('*.png')) + list(cdir.glob('*.JPG'))
        if imgs:
            sample_paths[cname] = imgs[0]

def build_figure(model, gradcam, class_names, all_class_images, fig=None):
    """
    Picks one random image per class, runs Grad-CAM, and (re)draws the
    full comparison figure onto `fig` (creates a new one if None).
    Returns the figure so the button callback can reuse it.
    """
    # pick fresh random samples every call
    sample_paths = {
        cname: random.choice(imgs)
        for cname, imgs in all_class_images.items() if imgs
    }

    n = len(sample_paths)

    if fig is None:
        fig = plt.figure(figsize=(17, 4.3 * n + 0.6))
    else:
        fig.clf()

    fig.patch.set_facecolor(BG_COLOR)

    # reserve a bit of bottom margin for the button
    gs = gridspec.GridSpec(n, 4, width_ratios=[1, 1, 1, 1.15],
                            hspace=0.55, wspace=0.12,
                            left=0.03, right=0.97, top=0.90, bottom=0.09,
                            figure=fig)

    def style_image_axis(ax, title, subtitle=None, accent=ACCENT):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        rect = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                               boxstyle="round,pad=0.01,rounding_size=0.03",
                               linewidth=1.4, edgecolor=accent, facecolor='none',
                               mutation_aspect=1, clip_on=False, zorder=10)
        ax.add_patch(rect)
        ax.text(0.5, 1.10, title, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=TEXT_MAIN)
        if subtitle:
            ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=9.5, color=TEXT_MUTED)

    for row, (cname, img_path) in enumerate(sample_paths.items()):
        image_pil = Image.open(img_path).convert('RGB')
        tensor, image_resized = preprocess_image(image_pil)

        cam, pred_idx, probs = gradcam.generate(tensor)
        overlay, heatmap_rgb = make_overlay_with_contour(image_resized, cam)

        pred_name = class_names[pred_idx]
        confidence = probs[pred_idx] * 100
        correct = (pred_name == cname)
        result_color = '#16a34a' if correct else '#dc2626'

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(image_resized)
        style_image_axis(ax0, "MRI Scan", f"True label: {cname}", accent='#2563eb')

        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(heatmap_rgb)
        style_image_axis(ax1, "Grad-CAM Heatmap", "Activation intensity", accent=ACCENT2)

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(overlay)
        style_image_axis(ax2, "Overlay + Contour", f"Pred: {pred_name}  ·  {confidence:.1f}%",
                          accent=result_color)
        ax2.text(0.5, -0.09, ("✓ Correct" if correct else "✗ Mismatch"),
                  transform=ax2.transAxes, ha='center', va='top',
                  fontsize=9, fontweight='bold', color=result_color)

        ax3 = fig.add_subplot(gs[row, 3])
        ax3.set_facecolor(PANEL_COLOR)
        colors = [BAR_TOP if i == pred_idx else BAR_OTHER for i in range(len(class_names))]
        y_pos = np.arange(len(class_names))
        ax3.barh(y_pos, probs * 100, color=colors, height=0.55, edgecolor='none', zorder=3)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(class_names, fontsize=9.5, color=TEXT_MAIN)
        ax3.set_xlim(0, 108)
        ax3.invert_yaxis()
        ax3.grid(axis='x', color=GRID_COLOR, linewidth=0.7, zorder=0)
        for spine in ax3.spines.values():
            spine.set_visible(False)
        ax3.tick_params(length=0)
        ax3.set_title("Class Probabilities", fontsize=12, fontweight='bold',
                       color=TEXT_MAIN, pad=14)
        for i, p in enumerate(probs * 100):
            label_color = TEXT_MAIN if i == pred_idx else TEXT_MUTED
            fw = 'bold' if i == pred_idx else 'normal'
            ax3.text(p + 2, i, f"{p:.1f}%", va='center', fontsize=9,
                     color=label_color, fontweight=fw)

    fig.suptitle("Grad-CAM Visualization Across the 4 Brain Tumor Classes",
                 fontsize=19, fontweight='bold', color=TEXT_MAIN, y=0.975)
    fig.text(0.5, 0.955, "Model interpretability via class-activation mapping (conv4 target layer)",
              ha='center', fontsize=11, color=TEXT_MUTED)

    # ---- "New random images" button ----
    btn_ax = fig.add_axes([0.40, 0.015, 0.20, 0.045])
    btn_ax.set_zorder(20)
    button = Button(btn_ax, '🔀  New Random Images',
                     color='#ede9fe', hovercolor='#ddd6fe')
    button.label.set_fontsize(11)
    button.label.set_fontweight('bold')
    button.label.set_color(ACCENT)
    for spine in btn_ax.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    def on_click(event):
        build_figure(model, gradcam, class_names, all_class_images, fig=fig)
        fig.canvas.draw_idle()

    button.on_clicked(on_click)
    fig._random_button_ref = button  # keep a reference so it isn't garbage-collected

    fig.canvas.draw_idle()
    return fig


def main():
    # ---- CONFIG: edit these paths ----
    MODEL_PATH = 'best_brain_tumor_model_patent.pth'
    SAMPLE_DIR = Path('sample_images')   # expects subfolders: glioma, meningioma, notumor, pituitary
    OUTPUT_PATH = 'gradcam_4classes.png'
    # -----------------------------------

    device = torch.device('cpu')

    model = UltraLightPatentBrainTumorCNN(num_classes=4).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])
    gradcam = GradCAM(model, target_layer=model.conv4)

    # collect ALL candidate images per class (not just one) so the button
    # can pick a fresh random one each time it's clicked
    all_class_images = {}
    for cname in class_names:
        cdir = SAMPLE_DIR / cname
        imgs = list(cdir.glob('*.jpg')) + list(cdir.glob('*.png')) + list(cdir.glob('*.JPG'))
        if imgs:
            all_class_images[cname] = imgs

    fig = build_figure(model, gradcam, class_names, all_class_images)

    # optional: save the first render to disk too
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight', facecolor=BG_COLOR)
    print(f"Saved initial figure to: {OUTPUT_PATH}")
    print("Click '🔀 New Random Images' in the window to resample images per class.")

    plt.show()


if __name__ == '__main__':
    main()