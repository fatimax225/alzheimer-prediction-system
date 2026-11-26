import os
import torch
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# ────────────────────────────────────────────────────────
# ⚙️ 1. إعدادات المسارات والبيانات
# ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# مسار مجلد البيانات الذي يحتوي على المجلدات الأربعة
DATA_DIR = os.path.join(BASE_DIR, "static", "dataset")

# التحقق من وجود المجلد
if not os.path.exists(DATA_DIR):
    raise Exception(f"⚠️ مجلد البيانات غير موجود في المسار: {DATA_DIR}\nيرجى التأكد من وضع مجلدات الصور الأربعة داخله.")

# ────────────────────────────────────────────────────────
# 🖼️ 2. توحيد حجم الصور وتحويلها
# ────────────────────────────────────────────────────────
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # توحيد مقاس جميع الصور
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ────────────────────────────────────────────────────────
# 📊 3. قراءة البيانات وتقسيمها
# ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("🚀 Loading Dataset...")
dataset = ImageFolder(DATA_DIR, transform=data_transform)
n = len(dataset)
n_test = int(0.2 * n)  # تخصيص 20% من الصور للاختبار
n_train = n - n_test

train_data, test_data = random_split(dataset, [n_train, n_test], torch.Generator().manual_seed(42))

# استخدام batch_size=32 لتسريع التدريب ومنع استهلاك الذاكرة
trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"✅ Found {n} images belonging to {len(dataset.classes)} classes.")
print(f"Classes dictionary: {dataset.class_to_idx}")
print("=" * 50 + "\n")


# ────────────────────────────────────────────────────────
# 🧠 4. بناء الشبكة العصبية (ConvNet)
# ────────────────────────────────────────────────────────
class ConvNet(torch.nn.Module):
    def __init__(self, num_classes=4):
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
        self.fc2 = torch.nn.Linear(32, num_classes)

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


# نقل المودل لمعالج الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}\n")

net = ConvNet(num_classes=len(dataset.classes)).to(device)

# ────────────────────────────────────────────────────────
# ⚙️ 5. إعدادات التدريب
# ────────────────────────────────────────────────────────
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.CrossEntropyLoss()
epochs = 20  # يمكنك زيادتها إلى 50 إذا أردت دقة أعلى لاحقاً

# ────────────────────────────────────────────────────────
# 🚀 6. حلقة التدريب والاختبار
# ────────────────────────────────────────────────────────
print("▶️ Starting Training...")
for epoch in range(epochs):
    net.train()
    running_loss_train = 0.0

    # تدريب المودل
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss_train += loss.item()

    avg_train_loss = running_loss_train / len(trainloader)

    # اختبار المودل في نفس الجولة
    net.eval()
    correct = 0
    total = 0
    running_loss_test = 0.0

    with torch.no_grad():  # إيقاف التعلم أثناء الاختبار
        for data in testloader:
            images, targets = data
            images, targets = images.to(device), targets.to(device)

            outputs = net(images)
            loss = loss_func(outputs, targets)
            running_loss_test += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_test_loss = running_loss_test / len(testloader)
    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2f}%")

print('\n🎉 Training and evaluation complete!')

# ────────────────────────────────────────────────────────
# 💾 7. حفظ المودل ليعمل مع موقعك
# ────────────────────────────────────────────────────────
model_path = os.path.join(BASE_DIR, "pytorch_alzheimer_model.pth")
torch.save(net.state_dict(), model_path)
print(f"✅ Model saved successfully at: {model_path}")
print("يمكنك الآن تشغيل السيرفر python3 app.py وتجربة الموقع!")