import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn


# 1. โครงสร้างโมเดล 2D-CNN
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


@st.cache_resource
def load_trained_model():
  model = AudioDeepfakeCNN()
  model.eval()
  return model


model = load_trained_model()

# 2. การตั้งค่าหน้าเว็บ Streamlit
st.title("Audio Deepfake Detection System")
st.write(
    "ระบบตรวจสอบและวิเคราะห์เสียงพูดระหว่าง **มนุษย์ (Real)** และ **AI สังเคราะห์"
    " (Fake)**"
)

# ค้นหาไฟล์ตัวอย่างใน dataset
real_folder = "dataset/real"
fake_folder = "dataset/fake"

real_files = (
    [os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.lower().endswith(".wav")]
    if os.path.exists(real_folder)
    else []
)
fake_files = (
    [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) if f.lower().endswith(".wav")]
    if os.path.exists(fake_folder)
    else []
)

st.sidebar.header("📁 เลือกแหล่งที่มาของไฟล์")
source_option = st.sidebar.radio(
    "เลือกรูปแบบการทดสอบ:", ("ไฟล์ตัวอย่างในระบบ", "อัปโหลดไฟล์เสียงของคุณเอง")
)

target_file = None

if source_option == "ไฟล์ตัวอย่างในระบบ":
  choice = st.sidebar.radio(
      "เลือกประเภทเสียง:",
      ("เสียงมนุษย์ (Real Sample)", "เสียง AI (Fake Sample)"),
  )
  if "Real" in choice and real_files:
    target_file = st.sidebar.selectbox("เลือกเสียงมนุษย์ (Real):", real_files)
  elif "Fake" in choice and fake_files:
    target_file = st.sidebar.selectbox("เลือกเสียง AI (Fake):", fake_files)
  else:
    st.sidebar.warning("ไม่พบไฟล์ในโฟลเดอร์ dataset")
else:
  uploaded_file = st.sidebar.file_uploader(
      "อัปโหลดไฟล์เสียง (.wav)", type=["wav"]
  )
  if uploaded_file is not None:
    temp_path = "temp_uploaded.wav"
    with open(temp_path, "wb") as f:
      f.write(uploaded_file.getbuffer())
    target_file = temp_path

if target_file and os.path.exists(target_file):
  st.markdown("---")
  st.subheader(f"🎧 ไฟล์ที่กำลังตรวจสอบ: `{os.path.basename(target_file)}`")

  # ปุ่มกดฟังเสียง
  st.audio(target_file, format="audio/wav")

  # โหลดสัญญาณเสียง
  y, sr = librosa.load(target_file, sr=16000, duration=10.0)

  # ทำนายผลด้วย AI
  mfccs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
  mfccs_db = librosa.power_to_db(mfccs)
  if mfccs_db.shape[1] < 200:
    mfccs_db = np.pad(
        mfccs_db, ((0, 0), (0, 200 - mfccs_db.shape[1])), mode="constant"
    )
  else:
    mfccs_db = mfccs_db[:, :200]

  input_tensor = (
      torch.tensor(mfccs_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
  )

  with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]
    pred_label = torch.argmax(probabilities).item()
    confidence = probabilities[pred_label].item() * 100

  # แสดงผล Prediction & Confidence
  col1, col2 = st.columns(2)
  with col1:
    if pred_label == 0:
      st.success("### ผลการทำนาย: เสียงมนุษย์ (REAL)")
    else:
      st.error("### ผลการทำนาย: เสียง AI (FAKE)")
  with col2:
    st.metric(
        label="ความมั่นใจของโมเดล (Confidence Score)",
        value=f"{confidence:.2f}%",
    )

  st.markdown("---")
  st.subheader("📊 เปรียบเทียบสเปกตรัมความถี่ (MFCC vs LFCC)")
  st.info(
      " กราฟ **Linear-Spectrogram (ด้านล่างสุด)** จะแบ่งสเกลความถี่เท่ากันตลอดช่วง"
      " ทำให้เราเห็น **ย่านความถี่สูง (High-Frequency)** ได้ชัดเจน ซึ่งเสียง AI"
      " มักจะมีความผิดปกติ เช่น ย่านความถี่สูงโล่งเตียนหรือตัดขาดอย่างไร้ธรรมชาติ"
  )

  # พล็อตราฟ 3 แบบ: Waveform, Mel-Spectrogram, Linear-Spectrogram
  fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 9))

  # 1. Waveform
  librosa.display.waveshow(y, sr=sr, ax=ax[0], color="blue")
  ax[0].set(title=f"Waveform (Time-Domain) - {os.path.basename(target_file)}")
  ax[0].set_xlabel("Time (s)")
  ax[0].set_ylabel("Amplitude")

  # 2. Mel-Spectrogram (Perceptual Scale)
  img1 = librosa.display.specshow(
      mfccs_db, sr=sr, x_axis="time", y_axis="mel", ax=ax[1], cmap="coolwarm"
  )
  ax[1].set(title="Mel-Spectrogram (Mel-Scale / MFCC-like)")
  fig.colorbar(img1, ax=ax[1], format="%+2.0f dB")

  # 3. Linear-Spectrogram (Linear-Scale / LFCC perspective)
  D = librosa.stft(y, n_fft=2048, hop_length=512)
  S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
  img2 = librosa.display.specshow(
      S_db,
      sr=sr,
      hop_length=512,
      x_axis="time",
      y_axis="linear",
      ax=ax[2],
      cmap="magma",
  )
  ax[2].set(
      title="Linear-Spectrogram (Linear-Scale / LFCC High-Frequency View)"
  )
  fig.colorbar(img2, ax=ax[2], format="%+2.0f dB")

  plt.tight_layout()
  st.pyplot(fig)

else:
  st.warning(
      "⚠️ กรุณาเลือกไฟล์ตัวอย่างจากในระบบ หรืออัปโหลดไฟล์เสียง .wav"
      " ที่แถบเมนูด้านซ้ายเพื่อเริ่มต้นใช้งาน"
  )