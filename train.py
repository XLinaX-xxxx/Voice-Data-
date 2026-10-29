import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# กำหนดค่าพารามิเตอร์มาตรฐาน
SAMPLE_RATE = 16000
N_FILTERS = 32
MAX_LEN = 200
FEATURE_TYPE = (  # เปลี่ยนเป็น "MFCC" หรือ "LFCC" ตามต้องการเพื่อเทรนโมเดลนั้นๆ
    "MFCC"
)


def get_linear_filterbank(sr, n_fft, n_filters=32):
  """สร้าง Linear-Scale Filter Bank สำหรับ LFCC ด้วย librosa"""
  min_freq = 0.0
  max_freq = sr / 2.0
  fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
  filter_pts = np.linspace(min_freq, max_freq, n_filters + 2)

  filterbank = np.zeros((n_filters, len(fft_freqs)))
  for i in range(1, n_filters + 1):
    f_prev = filter_pts[i - 1]
    f_curr = filter_pts[i]
    f_next = filter_pts[i + 1]

    filterbank[i - 1] = np.where(
        (fft_freqs >= f_prev) & (fft_freqs <= f_curr),
        (fft_freqs - f_prev) / (f_curr - f_prev + 1e-8),
        filterbank[i - 1],
    )
    filterbank[i - 1] = np.where(
        (fft_freqs >= f_curr) & (fft_freqs <= f_next),
        (f_next - fft_freqs) / (f_next - f_curr + 1e-8),
        filterbank[i - 1],
    )
  return filterbank


def extract_features(y, sr, feature_type="MFCC", max_len=200):
  """ฟังก์ชันสกัดฟีเจอร์ รองรับทั้ง MFCC (Mel-Scale) และ LFCC (Linear-Scale)"""
  n_fft = 2048
  hop_length = 512

  if feature_type == "MFCC":
    spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=32, n_fft=n_fft, hop_length=hop_length
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
  else:
    stft_matrix = (
        np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    )
    linear_fb = get_linear_filterbank(sr, n_fft, n_filters=32)
    spec = np.dot(linear_fb, stft_matrix)
    spec_db = librosa.power_to_db(spec, ref=np.max)

  current_len = spec_db.shape[1]
  if current_len < max_len:
    pad_width = max_len - current_len
    spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode="constant")
  else:
    spec_db = spec_db[:, :max_len]

  return spec_db


def apply_augmentation(y):
  """เพิ่ม Data Augmentation เพื่อให้โมเดลเรียนรู้ความหลากหลายของเสียง"""
  # 1. สุ่มเติม White Noise เบาๆ (ช่วยให้โมเดลไม่ติดความเงียบกริ๊บ)
  if np.random.rand() < 0.5:
    noise = np.random.randn(len(y))
    y = y + 0.005 * noise

  # 2. สุ่มปรับความดัง (Volume Perturbation)
  if np.random.rand() < 0.5:
    factor = np.random.uniform(0.8, 1.2)
    y = y * factor

  return y


class AudioDataset(Dataset):

  def __init__(self, data_dir, feature_type="MFCC", max_len=200, augment=True):
    self.file_list = []
    self.labels = []
    self.feature_type = feature_type
    self.max_len = max_len
    self.augment = augment  # เปิดใช้งาน Data Augmentation เฉพาะตอนเทรน

    real_dir = os.path.join(data_dir, "real")
    if os.path.exists(real_dir):
      for f in os.listdir(real_dir):
        if f.lower().endswith((".flac", ".wav", ".mp3")):
          self.file_list.append(os.path.join(real_dir, f))
          self.labels.append(0)

    fake_dir = os.path.join(data_dir, "fake")
    if os.path.exists(fake_dir):
      for f in os.listdir(fake_dir):
        if f.lower().endswith((".flac", ".wav", ".mp3")):
          self.file_list.append(os.path.join(fake_dir, f))
          self.labels.append(1)

    print(
        f"--- โหลดข้อมูลสำเร็จ [{feature_type}]: พบไฟล์เสียงทั้งหมด"
        f" {len(self.file_list):,} ไฟล์ ---"
    )

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, idx):
    while True:
      audio_path = self.file_list[idx]
      try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=4.0)

        # ทำ Data Augmentation เฉพาะตอนฝึกซ้อม
        if self.augment:
          y = apply_augmentation(y)

        feat_db = extract_features(
            y, sr, feature_type=self.feature_type, max_len=self.max_len
        )

        feature_tensor = torch.tensor(feat_db, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature_tensor, label_tensor

      except Exception:
        idx = (idx + 1) % len(self.file_list)


class AudioDeepfakeCNN(nn.Module):

  def __init__(self):
    super(AudioDeepfakeCNN, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    # เพิ่ม Dropout เพื่อป้องกัน Overfitting (สุ่มปิด 30% ระหว่างเทรน)
    self.fc1 = nn.Linear(32 * 8 * 50, 64)
    self.dropout = nn.Dropout(0.3)
    self.fc2 = nn.Linear(64, 2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    x = torch.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x


if __name__ == "__main__":
  print(f"--- เริ่มต้นกระบวนการเทรนด้วยฟีเจอร์: {FEATURE_TYPE} ---")
  dataset = AudioDataset(
      "dataset", feature_type=FEATURE_TYPE, max_len=MAX_LEN, augment=True
  )

  if len(dataset) == 0:
    print("❌ ไม่พบไฟล์เสียงในโฟลเดอร์ dataset!")
  else:
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioDeepfakeCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 15  # เพิ่มจำนวนรอบขึ้นอีกนิดเพื่อให้โมเดลเรียนรู้ได้ลึกขึ้น
    for epoch in range(epochs):
      model.train()
      running_loss = 0.0
      correct = 0
      total = 0

      for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

      acc = 100 * correct / total
      print(
          f"Epoch [{epoch+1}/{epochs}], Loss:"
          f" {running_loss/len(dataloader):.4f}, Accuracy: {acc:.2f}%"
      )

    model_filename = (
        "best_model_lfcc.pth" if FEATURE_TYPE == "LFCC" else "best_model_mfcc.pth"
    )
    torch.save(model.state_dict(), model_filename)
    print(
        f"--- เทรนเสร็จสิ้น และบันทึกโมเดลลงในเครื่องเรียบร้อยแล้ว ({model_filename})"
        " ---"
    )