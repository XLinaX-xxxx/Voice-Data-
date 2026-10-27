import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class AudioDataset(Dataset):

  def __init__(self, data_dir, max_len=200):
    self.file_list = []
    self.labels = []
    self.max_len = max_len

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
        f"--- โหลดข้อมูลสำเร็จ: พบไฟล์เสียงทั้งหมด {len(self.file_list):,} ไฟล์ ---"
    )

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, idx):
    while True:
      audio_path = self.file_list[idx]
      try:
        y, sr = librosa.load(audio_path, sr=16000, duration=4.0)
        mfccs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
        mfccs = librosa.power_to_db(mfccs)

        if mfccs.shape[1] < self.max_len:
          mfccs = np.pad(
              mfccs, ((0, 0), (0, self.max_len - mfccs.shape[1])), mode="constant"
          )
        else:
          mfccs = mfccs[:, : self.max_len]

        feature_tensor = torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0)
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
    self.fc1 = nn.Linear(32 * 8 * 50, 64)
    self.fc2 = nn.Linear(64, 2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x


if __name__ == "__main__":
  dataset = AudioDataset("dataset", max_len=200)

  if len(dataset) == 0:
    print("❌ ไม่พบไฟล์เสียงในโฟลเดอร์ dataset!")
  else:
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    model = AudioDeepfakeCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    print("--- เริ่มต้นเทรน AI (Audio Deepfake Detection) ---")
    epochs = 10

    for epoch in range(epochs):
      model.train()
      running_loss = 0.0
      correct = 0
      total = 0

      for inputs, labels in dataloader:
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
          f" {running_loss/len(dataloader):.4f}, ความแม่นยำ (Accuracy):"
          f" {acc:.2f}%"
      )

    torch.save(model.state_dict(), "best_model.pth")
    print(
        "--- เทรนเสร็จสิ้น และบันทึกโมเดลลงใน 'best_model.pth' เรียบร้อยแล้ว! ---"
    )