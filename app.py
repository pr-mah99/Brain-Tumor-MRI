import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import os
import time
from pathlib import Path

# ======================== Model Components ========================
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
        x_spatial = x * spatial_weights
        channel_pool = self.channel_avg(x_spatial)
        channel_weights = torch.sigmoid(self.channel_out(F.relu(self.channel_fc(channel_pool))))
        x_enhanced = x_spatial * channel_weights
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
        decision_ch = max(2, channels // 8)
        self.decision = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, decision_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decision_ch, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        decision_weights = self.decision(x)
        max_pool = F.max_pool2d(x, kernel_size=2, stride=2)
        avg_pool = F.avg_pool2d(x, kernel_size=2, stride=2)
        w1 = decision_weights[:, 0:1, :, :].view(B, 1, 1, 1)
        w2 = decision_weights[:, 1:2, :, :].view(B, 1, 1, 1)
        return w1 * max_pool + w2 * avg_pool

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

# ======================== Flask App ========================
app = Flask(__name__)
device = torch.device('cpu')

# تحميل النموذج
model = UltraLightPatentBrainTumorCNN(num_classes=4).to(device)
checkpoint = torch.load('best_brain_tumor_model_patent.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])

# ======================== تحميل الصور من المجلد ========================
def load_sample_images():
    sample_dir = Path('sample_images')
    sample_images = {}
    
    for class_name in class_names:
        class_dir = sample_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.JPG'))
            sample_images[class_name] = images[:20]  # أقصى 20 صورة
    
    return sample_images

sample_images = load_sample_images()

def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تصنيف أورام الدماغ</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 25px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.2);
            max-width: 900px;
            width: 100%;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 28px;
            margin-bottom: 8px;
            font-weight: 700;
        }
        
        .header p {
            font-size: 13px;
            opacity: 0.9;
        }
        
        .content {
            padding: 35px;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .tab-btn {
            padding: 10px 20px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            color: #999;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        
        .tab-btn.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .upload-zone {
            border: 2px dashed #667eea;
            border-radius: 16px;
            padding: 35px;
            text-align: center;
            cursor: pointer;
            background: linear-gradient(135deg, #f8f9ff 0%, #eff1ff 100%);
            transition: all 0.3s ease;
        }
        
        .upload-zone:hover {
            border-color: #764ba2;
            background: linear-gradient(135deg, #eff1ff 0%, #e6e9ff 100%);
        }
        
        .upload-zone input { display: none; }
        .upload-icon { font-size: 50px; margin-bottom: 12px; }
        .upload-text { color: #667eea; font-weight: 600; font-size: 16px; margin-bottom: 4px; }
        .upload-subtext { color: #999; font-size: 12px; }
        
        .samples-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .sample-item {
            cursor: pointer;
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid transparent;
            transition: all 0.3s;
            background: #f5f5f5;
        }
        
        .sample-item:hover {
            border-color: #667eea;
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        }
        
        .sample-item img {
            width: 100%;
            height: 120px;
            object-fit: cover;
        }
        
        .preview { display: none; margin-bottom: 25px; position: relative; }
        .preview img { width: 100%; max-height: 300px; object-fit: contain; border-radius: 12px; display: block; }
        
        .preview-container {
            position: relative;
            width: 100%;
            display: inline-block;
        }
        
        .controls {
            display: none;
            gap: 10px;
            margin-bottom: 25px;
        }
        
        button {
            flex: 1;
            padding: 13px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            font-size: 15px;
            transition: all 0.3s;
        }
        
        .btn-classify {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .btn-classify:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-clear {
            background: #f5f5f5;
            color: #667eea;
            border: 1px solid #e0e0e0;
        }
        
        .loading { display: none; text-align: center; padding: 25px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 45px; height: 45px; animation: spin 0.8s linear infinite; margin: 0 auto 12px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .results {
            display: none;
            background: linear-gradient(135deg, #f8f9ff 0%, #eff1ff 100%);
            border-radius: 15px;
            padding: 28px;
            border-left: 5px solid #667eea;
        }
        
        .result-value { font-size: 32px; font-weight: 800; color: #667eea; margin-bottom: 15px; }
        .conf-bar { height: 8px; background: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .conf-fill { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }
        
        .score-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            font-size: 13px;
        }
        
        .latency-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: white;
            border: 1px solid #dfe3ff;
            color: #667eea;
            font-size: 12px;
            font-weight: 700;
            padding: 6px 12px;
            border-radius: 20px;
            margin-top: 12px;
        }
        
        .latency-badge .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4caf50;
        }
        
        .latency-badge.slow .dot { background: #ff9800; }
        .latency-badge.verySlow .dot { background: #e53935; }
        
        .error { display: none; background: #ffebee; color: #c62828; padding: 14px; border-radius: 10px; border-left: 4px solid #c62828; }
        
        .ai-scanner {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        
        .scanner-box {
            position: relative;
            width: 200px;
            height: 200px;
            border: 3px solid #667eea;
            border-radius: 20px;
            background: rgba(102, 126, 234, 0.1);
            overflow: hidden;
        }
        
        .scan-line {
            position: absolute;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
            top: 0;
            animation: scanMove 1.5s ease-in-out infinite;
        }
        
        @keyframes scanMove { 0% { top: 0; } 50% { top: 197px; } 100% { top: 197px; } }
        
        .pulse {
            animation: pulse 0.8s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
        
        .corner { position: absolute; width: 20px; height: 20px; border: 2px solid #667eea; }
        .corner-tl { top: -3px; left: -3px; border-right: none; border-bottom: none; }
        .corner-tr { top: -3px; right: -3px; border-left: none; border-bottom: none; }
        .corner-bl { bottom: -3px; left: -3px; border-right: none; border-top: none; }
        .corner-br { bottom: -3px; right: -3px; border-left: none; border-top: none; }
        
        .scanner-text {
            position: absolute;
            bottom: -30px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            font-size: 14px;
            font-weight: 600;
            white-space: nowrap;
        }
        
        .scroll-up {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            font-size: 24px;
            display: none;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s;
            z-index: 999;
        }
        
        .scroll-up:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 تصنيف أورام الدماغ</h1>
            <p>Brain Tumor Classification System</p>
        </div>
        
        <div class="content">
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('upload')">رفع صورة</button>
                <button class="tab-btn" onclick="switchTab('samples')">صور عينات</button>
            </div>
            
            <!-- تبويب الرفع اليدوي -->
            <div class="tab-content active" id="upload">
                <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                    <input type="file" id="fileInput" accept="image/*">
                    <div class="upload-icon">📤</div>
                    <div class="upload-text">اضغط لرفع صورة MRI</div>
                    <div class="upload-subtext">أو اسحب الصورة هنا</div>
                </div>
                
                <div class="preview-container">
                    <div class="preview" id="preview">
                        <img id="previewImg" src="" alt="Preview">
                    </div>
                </div>
                
                <div class="controls" id="controls">
                    <button class="btn-classify" onclick="classify()">تصنيف الصورة</button>
                    <button class="btn-clear" onclick="clearImage()">مسح</button>
                </div>
            </div>
            
            <!-- تبويب الصور العينات -->
            <div class="tab-content" id="samples">
                <div id="samplesContainer"></div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div>جاري التحليل...</div>
            </div>
            
            <div class="results" id="results">
                <div class="result-value" id="resultValue">-</div>
                <div>
                    <strong>درجة الثقة:</strong>
                    <div class="conf-bar"><div class="conf-fill" id="confBar"></div></div>
                    <div id="confText">0%</div>
                </div>
                <div id="allScores" style="margin-top: 15px;"></div>
                <div id="latencyBadge" class="latency-badge" style="display:none;">
                    <span class="dot"></span>
                    <span id="latencyText">0 ms</span>
                </div>
            </div>
            
            <div class="error" id="error"></div>
        </div>
    </div>
    

    
    <button class="scroll-up" id="scrollUp" onclick="scrollToTop()">⬆️</button>
    
    <script>
        let selectedFile = null;
        
        function switchTab(tab) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(tab).classList.add('active');
            if(event && event.target) event.target.classList.add('active');
            else document.querySelector('[onclick*="' + tab + '"]').classList.add('active');
        }
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    document.getElementById('previewImg').src = event.target.result;
                    document.getElementById('preview').style.display = 'block';
                    document.getElementById('controls').style.display = 'flex';
                };
                reader.readAsDataURL(selectedFile);
            }
        });
        
        function loadSampleImages(data) {
            const container = document.getElementById('samplesContainer');
            container.innerHTML = '';
            
            data.samples.forEach(category => {
                const section = document.createElement('div');
                section.style.marginBottom = '30px';
                
                const title = document.createElement('h3');
                title.textContent = category.class_name;
                title.style.marginBottom = '10px';
                title.style.color = '#667eea';
                section.appendChild(title);
                
                const grid = document.createElement('div');
                grid.className = 'samples-grid';
                
                category.images.forEach(imgData => {
                    const item = document.createElement('div');
                    item.className = 'sample-item';
                    item.innerHTML = '<img src="' + imgData.src + '" alt="sample">';
                    item.onclick = () => selectSampleImage(imgData.path);
                    grid.appendChild(item);
                });
                
                section.appendChild(grid);
                container.appendChild(section);
            });
        }
        
        function selectSampleImage(imagePath) {
            fetch('/get_image', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: imagePath})
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('previewImg').src = 'data:image/jpeg;base64,' + data.image;
                document.getElementById('preview').style.display = 'block';
                selectedFile = {path: imagePath};
                document.getElementById('controls').style.display = 'flex';
                document.getElementById('results').style.display = 'none';
                
                // الانتقال التلقائي لتبويب رفع الصورة
                switchTab('upload');
                setTimeout(() => {
                    document.getElementById('previewImg').scrollIntoView({behavior: 'smooth'});
                }, 100);
            });
        }
        
        function classify() {
            if (!selectedFile) return;
            
            const formData = new FormData();
            if (selectedFile.path) {
                formData.append('sample_path', selectedFile.path);
            } else {
                formData.append('image', selectedFile);
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            const clientStart = performance.now();
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                const clientEnd = performance.now();
                const roundTripMs = (clientEnd - clientStart).toFixed(1);
                
                document.getElementById('loading').style.display = 'none';
                if (data.error) {
                    document.getElementById('error').textContent = 'خطأ: ' + data.error;
                    document.getElementById('error').style.display = 'block';
                } else {
                    document.getElementById('resultValue').textContent = data.class_name;
                    const conf = data.confidence * 100;
                    document.getElementById('confBar').style.width = conf + '%';
                    document.getElementById('confText').textContent = conf.toFixed(1) + '%';
                    
                    let html = '';
                    data.all_scores.forEach((score, i) => {
                        const pct = (score * 100).toFixed(1);
                        html += '<div class="score-row"><span>' + data.class_names[i] + '</span><span>' + pct + '%</span></div>';
                    });
                    document.getElementById('allScores').innerHTML = html;
                    
                    // عرض زمن الاستدلال
                    const inferenceMs = data.inference_ms;
                    const preprocessMs = data.preprocess_ms;
                    const totalServerMs = data.total_ms;
                    
                    const badge = document.getElementById('latencyBadge');
                    badge.classList.remove('slow', 'verySlow');
                    if (inferenceMs > 500) badge.classList.add('verySlow');
                    else if (inferenceMs > 150) badge.classList.add('slow');
                    
                    document.getElementById('latencyText').textContent =
                        'الاستدلال: ' + inferenceMs.toFixed(1) + ' ms' +
                        '  |  المعالجة: ' + preprocessMs.toFixed(1) + ' ms' +
                        '  |  إجمالي السيرفر: ' + totalServerMs.toFixed(1) + ' ms' +
                        '  |  زمن الشبكة الكلي: ' + roundTripMs + ' ms';
                    badge.style.display = 'inline-flex';
                    
                    document.getElementById('results').style.display = 'block';
                    
                    // Scroll للنتيجة
                    setTimeout(() => {
                        document.getElementById('results').scrollIntoView({behavior: 'smooth', block: 'center'});
                        document.getElementById('scrollUp').style.display = 'flex';
                    }, 300);
                }
            });
        }
        
        function scrollToTop() {
            window.scrollTo({top: 0, behavior: 'smooth'});
        }
        
        window.addEventListener('scroll', () => {
            if (window.scrollY > 300) {
                document.getElementById('scrollUp').style.display = 'flex';
            } else {
                document.getElementById('scrollUp').style.display = 'none';
            }
        });
        
        function clearImage() {
            selectedFile = null;
            document.getElementById('preview').style.display = 'none';
            document.getElementById('controls').style.display = 'none';
            document.getElementById('results').style.display = 'none';
        }
        
        // تحميل الصور عند بدء الصفحة
        fetch('/get_samples').then(r => r.json()).then(loadSampleImages);
    </script>
</body>
</html>
'''

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
                
                # تحديد نوع الصورة تلقائياً
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
        request_start = time.perf_counter()

        if 'sample_path' in request.form:
            image = Image.open(request.form['sample_path']).convert('RGB')
        else:
            image = Image.open(BytesIO(request.files['image'].read())).convert('RGB')

        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = img_tensor.to(device)

        preprocess_end = time.perf_counter()

        # قياس زمن الاستدلال الفعلي فقط (forward pass)
        inference_start = time.perf_counter()
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0].cpu().numpy()
        inference_end = time.perf_counter()

        pred_idx = np.argmax(probabilities)

        request_end = time.perf_counter()

        preprocess_ms = (preprocess_end - request_start) * 1000
        inference_ms = (inference_end - inference_start) * 1000
        total_ms = (request_end - request_start) * 1000

        return jsonify({
            'class_name': class_names[pred_idx],
            'confidence': float(probabilities[pred_idx]),
            'all_scores': probabilities.tolist(),
            'class_names': class_names,
            'preprocess_ms': preprocess_ms,
            'inference_ms': inference_ms,
            'total_ms': total_ms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)