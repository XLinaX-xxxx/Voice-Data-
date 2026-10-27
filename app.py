import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

# กำหนดนามสกุลไฟล์เสียงที่รองรับทั้งหมด
SUPPORTED_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")

# กำหนดค่ามาตรฐานในการสกัดฟีเจอร์ให้ตรงกับตอนเทรนโมเดล
SAMPLE_RATE = 16000
N_MELS = 32
MAX_LEN = 200


# 1. โครงสร้างโมเดล 2D-CNN (ตรงกับโครงสร้างจริงในไฟล์น้ำหนัก)
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
    # คำนวณ Fully Connected 
    # n_mels=32 ผ่าน MaxPool 2 ครั้งจะเหลือ 32 / 4 = 8
    # max_len=200 ผ่าน MaxPool 2 ครั้งจะเหลือ 200 / 4 = 50 -> 32 * 8 * 50 = 12800
    self.fc1 = nn.Linear(32 * 8 * 50, 64)
    self.fc2 = nn.Linear(64, 2)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x


# โหลดโมเดลที่เทรนแล้วพร้อมระบบตรวจสอบข้อผิดพลาด
@st.cache_resource
def load_trained_model():
  model_path = "best_model.pth"
  if not os.path.exists(model_path):
    return None

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AudioDeepfakeCNN().to(device)

  try:
    checkpoint = torch.load(model_path, map_location=device)
    # รองรับทั้งแบบบันทึก state_dict ตรงๆ หรือบันทึกเก็บไว้ใน dictionary
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
      model.load_state_dict(checkpoint["state_dict"])
    else:
      model.load_state_dict(checkpoint)
    model.eval()
    return model, device
  except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดโครงสร้างน้ำหนักโมเดล: {e}")
    return None, device


# 2. การตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="Audio Deepfake Detection", page_icon="🎙️", layout="wide"
)

st.title("🎙️ Audio Deepfake Detection System")
st.write(
    "ระบบตรวจสอบและวิเคราะห์เสียงพูดเพื่อจำแนกเสียงระหว่าง **มนุษย์จริง (Real)**"
    " และ **เสียงสังเคราะห์จาก AI (Fake)**"
)

# โหลดโมเดล
model, device = load_trained_model()

if model is None:
  st.error(
      "⚠️ ไม่พบไฟล์น้ำหนักโมเดล `best_model.pth` หรือไฟล์เสียหาย"
      " กรุณาตรวจสอบว่ามีไฟล์ดังกล่าวอยู่ในโฟลเดอร์โปรเจกต์"
  )
else:
  # ค้นหาไฟล์ตัวอย่างใน dataset
  real_folder = "dataset/real"
  fake_folder = "dataset/fake"

  real_files = (
      [
          os.path.join(real_folder, f)
          for f in os.listdir(real_folder)
          if f.lower().endswith(SUPPORTED_EXTENSIONS)
      ]
      if os.path.exists(real_folder)
      else []
  )
  fake_files = (
      [
          os.path.join(fake_folder, f)
          for f in os.listdir(fake_folder)
          if f.lower().endswith(SUPPORTED_EXTENSIONS)
      ]
      if os.path.exists(fake_folder)
      else []
  )

  st.sidebar.header("📁 แหล่งที่มาของไฟล์เสียง")
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
      st.sidebar.warning(
          "ไม่พบไฟล์เสียงในโฟลเดอร์ dataset (ตรวจสอบว่ามีโฟลเดอร์ dataset/real"
          " และ dataset/fake หรือไม่)"
      )
  else:
    uploaded_file = st.sidebar.file_uploader(
        "อัปโหลดไฟล์เสียง (.wav, .flac, .mp3, .m4a, .ogg)",
        type=["wav", "flac", "mp3", "m4a", "ogg"],
    )
    if uploaded_file is not None:
      file_ext = os.path.splitext(uploaded_file.name)[1].lower()
      temp_path = f"temp_uploaded{file_ext}"

      with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
      target_file = temp_path

  if target_file and os.path.exists(target_file):
    st.markdown("---")
    st.subheader(f"🎧 ไฟล์ที่กำลังตรวจสอบ: `{os.path.basename(target_file)}`")

    st.audio(target_file)

    try:
      # โหลดไฟล์เสียงและกำหนด Sampling Rate เป็น 16kHz
      y, sr = librosa.load(target_file, sr=SAMPLE_RATE, duration=4.0)

      # แปลงเป็น Mel-Spectrogram ด้วยพารามิเตอร์ที่เสถียร
      mel_spec = librosa.feature.melspectrogram(
          y=y, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512
      )
      mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

      # ทำ Padding หรือ Truncating ให้มีความกว้างพอดีกับ MAX_LEN (200 เฟรม)
      current_len = mel_spec_db.shape[1]
      if current_len < MAX_LEN:
        pad_width = MAX_LEN - current_len
        mel_spec_db = np.pad(
            mel_spec_db, ((0, 0), (0, pad_width)), mode="constant"
        )
      else:
        mel_spec_db = mel_spec_db[:, :MAX_LEN]

      # แปลงเป็น Tensor และส่งไปยัง Device (CPU/GPU)
      input_tensor = (
          torch.tensor(mel_spec_db, dtype=torch.float32)
          .unsqueeze(0)
          .unsqueeze(0)
          .to(device)
      )

      # ทำนายผล
      with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        pred_label = torch.argmax(probabilities).item()
        confidence = probabilities[pred_label].item() * 100

      # แสดงผลลัพธ์ผ่านหน้าจอ
      col1, col2 = st.columns(2)
      with col1:
        if pred_label == 0:
          st.success("### ผลการทำนาย: เสียงมนุษย์จริง (REAL)")
        else:
          st.error("### ผลการทำนาย: เสียงปลอม/AI (FAKE)")
      with col2:
        st.metric(
            label="ความมั่นใจของโมเดล (Confidence Score)",
            value=f"{confidence:.2f}%",
        )

      st.markdown("---")
      st.subheader("📊 วิเคราะห์คลื่นความถี่และสเปกตรัมเสียง")

      fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 9))

      # 1. Waveform
      librosa.display.waveshow(y, sr=sr, ax=ax[0], color="royalblue")
      ax[0].set(
          title=f"Waveform (Time-Domain) - {os.path.basename(target_file)}"
      )
      ax[0].set_xlabel("Time (s)")
      ax[0].set_ylabel("Amplitude")

      # 2. Mel-Spectrogram
      img1 = librosa.display.specshow(
          mel_spec_db,
          sr=sr,
          x_axis="time",
          y_axis="mel",
          ax=ax[1],
          cmap="coolwarm",
      )
      ax[1].set(title="Mel-Spectrogram (Mel-Scale)")
      fig.colorbar(img1, ax=ax[1], format="%+2.0f dB")

      # 3. Linear-Spectrogram
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
      ax[2].set(title="Linear-Spectrogram (High-Frequency View)")
      fig.colorbar(img2, ax=ax[2], format="%+2.0f dB")

      plt.tight_layout()
      st.pyplot(fig)

    except Exception as e:
      st.error(
          f"⚠️ เกิดข้อผิดพลาดในการประมวลผลไฟล์เสียง: {e}"
          " (โปรดตรวจสอบไฟล์เสียงว่าอยู่ในรูปแบบที่รองรับและไม่เสียหาย)"
      )

  else:
    st.warning(
        "⚠️ กรุณาเลือกไฟล์ตัวอย่างจากในระบบ หรืออัปโหลดไฟล์เสียงผ่านเมนูด้านซ้าย"
        " เพื่อเริ่มการตรวจสอบ"
    )