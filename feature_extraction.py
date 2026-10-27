import torch
import torch.nn as nn
import torch.optim as optim


# 1. ออกแบบโครงสร้างโมเดล 2D-CNN สำหรับจำแนกเสียงจริง/ปลอม
class AudioDeepfakeCNN(nn.Module):

  def __init__(self):
    super(AudioDeepfakeCNN, self).__init__()
    # Convolutional Block 1
    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    # Convolutional Block 2
    self.conv2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    # Fully Connected Layers สำหรับตัดสินใจ (0 = Bonafide, 1 = Spoof)
    self.fc1 = nn.Linear(
        32 * 5 * 5, 64
    )  # ปรับขนาดตามมิติฟีเจอร์ที่ส่งเข้ามา
    self.fc2 = nn.Linear(64, 2)

  def forward(self, x):
    # x shape: (Batch_size, Channels, Height, Width)
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)  # Flatten
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x


# 2. จำลองการนำฟีเจอร์เข้าโมเดล (Training Loop Concept)
model = AudioDeepfakeCNN()
criterion = nn.CrossEntropyLoss()  # ฟังก์ชันคำนวณความผิดพลาด (Loss)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model Architecture Initialized Successfully!")