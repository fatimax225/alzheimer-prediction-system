import os
import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ────────────────────────────────────────────────────────
# 🤖 1. تحميل نماذج المشروع الأول (البيانات النصية Tabular)
# ────────────────────────────────────────────────────────
try:
    model_tabular = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler_tabular = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    print("✓ Tabular models loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning loading tabular models: {e}")


# ────────────────────────────────────────────────────────
# 🧠 2. بناء كلاس الشبكة العصبية (نفس الكود الذي تدرب عليه المودل)
# ────────────────────────────────────────────────────────
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2, 2))
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2, 2))
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2, 2))
        self.conv4 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2, 2))
        self.dp = torch.nn.Dropout(p=0.5)
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(128 * 8 * 8, 32), torch.nn.ReLU())
        self.fc2 = torch.nn.Linear(32, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dp(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out


# ────────────────────────────────────────────────────────
# 🧠 3. تحميل موديل PyTorch المُدرب للصور
# ────────────────────────────────────────────────────────
try:
    model_mri = ConvNet()
    # نستخدم map_location='cpu' ليعمل الموقع على أي جهاز حتى بدون كرت شاشة
    model_mri.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "pytorch_alzheimer_model.pth"), map_location=torch.device('cpu')))
    model_mri.eval()  # وضع التقييم (مهم جداً)

    # 💡 الكلاسات تترتب أبجدياً حسب أسماء المجلدات في تدريب PyTorch
    mri_classes = {
        0: "Mild Demented",
        1: "Moderate Demented",
        2: "Non-Demented (Normal)",
        3: "Very Mild Demented"
    }
    print("✓ PyTorch MRI Image model loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning loading PyTorch MRI model: {e}")

# إعدادات معالجة الصورة قبل إدخالها للمودل
inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# إعدادات لإنشاء الصور المائلة (للعرض في الموقع فقط)
display_aug_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.Resize((128, 128))
])


# ────────────────────────────────────────────────────────
# 🌐 4. الصفحات والمشروع الأول
# ────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/dashboard')
@app.route('/dashboard.html')
def dashboard_page():
    return render_template('dashboard.html')


@app.route('/assessment.html')
@app.route('/assessment')
def assessment_page():
    return render_template('assessment.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form.get('age'))
        mmse = float(request.form.get('mmse'))
        cdr = float(request.form.get('cdr'))
        gender = int(request.form.get('gender'))

        new_patient = pd.DataFrame([{"Age": age, "MMSE": mmse, "CDR": cdr, "M/F": gender}])
        new_patient = new_patient[["Age", "MMSE", "CDR", "M/F"]]

        new_patient_scaled = scaler_tabular.transform(new_patient)
        prediction = model_tabular.predict(new_patient_scaled)[0]

        return render_template("dashboard.html", result=str(prediction), age=int(age), mmse=int(mmse), cdr=cdr,
                               gender=gender)
    except Exception as e:
        return f"حدث خطأ أثناء معالجة البيانات: {str(e)}", 400


# ────────────────────────────────────────────────────────
# 🖼️ 5. المشروع الثاني (التنبؤ بصور الرنين - PyTorch)
# ────────────────────────────────────────────────────────
@app.route('/predict_mri', methods=['POST'])
def predict_mri():
    try:
        if 'mri_image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image uploaded'}), 400

        file = request.files['mri_image']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        upload_path = os.path.join(BASE_DIR, 'static', filename)
        file.save(upload_path)

        # 1. قراءة الصورة باستخدام PIL وتحويلها لـ RGB
        img = Image.open(upload_path).convert('RGB')

        # 2. 🌟 توليد صور طبية معدلة (للعرض في الموقع) 🌟
        aug_paths = []
        for i in range(3):
            aug_img = display_aug_transform(img)
            aug_filename = f'aug_{i}_{filename}'
            aug_path = os.path.join(BASE_DIR, 'static', aug_filename)
            aug_img.save(aug_path)
            aug_paths.append(aug_filename)

        # 3. تجهيز الصورة وفحصها بالمودل
        img_tensor = inference_transform(img).unsqueeze(0)  # إضافة بُعد الدفعة (Batch dimension)

        with torch.no_grad():  # إيقاف التدريب وتسريع الفحص
            outputs = model_mri(img_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        # استخراج النتيجة
        final_result = mri_classes.get(class_idx, "Unknown")

        return render_template("dashboard.html", mri_result=final_result, original_img=filename, aug_imgs=aug_paths)

    except Exception as e:
        return f"حدث خطأ أثناء معالجة الصورة: {str(e)}", 400


# تأكدي من استيراد nltk في بداية الملف
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

project_info = {
    "model": "We used VGG16, which is a powerful CNN (Convolutional Neural Network) for medical imaging.",
    "dataset": "The dataset contains MRI images balanced with 500 images per class.",
    "accuracy": "The model reached an accuracy of over 85% during training.",
    "flask": "Flask is used as a web framework to provide a user-friendly interface for the doctors.",
    "preprocessing": "We used Gray scaling and Thresholding to highlight the brain features before prediction."
}


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message", "").lower()
    tokens = word_tokenize(user_input)

    reply = "I'm sorry, I only know about the Alzheimer project. Ask about (model, accuracy, dataset, or preprocessing)."

    for word in tokens:
        if word in project_info:
            reply = project_info[word]
            break

    return {"reply": reply}


if __name__ == '__main__':
    app.run(debug=True)