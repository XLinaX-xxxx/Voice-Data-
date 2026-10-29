import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

SUPPORTED_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
SAMPLE_RATE = 16000
N_FILTERS = 32
MAX_LEN = 200


def get_linear_filterbank(sr, n_fft, n_filters=32):
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
  n_fft = 2048
  hop_length = 512

  if feature_type == "MFCC":
    spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=32, n_fft=n_fft, hop_length=hop_length
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
  else:
    stft_matrix = np.abs(
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    ) ** 2
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
def load_trained_model(feature_type):
  model_path = (
      "best_model_lfcc.pth" if feature_type == "LFCC" else "best_model_mfcc.pth"
  )
  if not os.path.exists(model_path):
    if os.path.exists("best_model.pth"):
      model_path = "best_model.pth"
    else:
      return None, None

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AudioDeepfakeCNN().to(device)

  try:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
      model.load_state_dict(checkpoint["state_dict"])
    else:
      model.load_state_dict(checkpoint)
    model.eval()
    return model, device
  except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดน้ำหนักโมเดล: {e}")
    return None, device


st.set_page_config(
    page_title="Audio Deepfake Detection", page_icon="🎙️", layout="wide"
)

st.title("🎙️ Audio Deepfake Detection System")
st.write(
    "ระบบตรวจสอบและวิเคราะห์เสียงพูดเพื่อจำแนกเสียงระหว่าง **มนุษย์จริง (Real)**"
    " และ **เสียงสังเคราะห์จาก AI (Fake)** ด้วยการเปรียบเทียบฟีเจอร์ MFCC และ"
    " LFCC"
)

# Sidebar เลือกประเภทฟีเจอร์
st.sidebar.header("⚙️ ตั้งค่าระบบประมวลผลสัญญาณ")
feature_choice = st.sidebar.selectbox(
    "เลือกฟีเจอร์ทางสัญญาณ (Feature Type):",
    ("LFCC (Linear-Scale)", "MFCC (Mel-Scale)"),
)
selected_feature = "LFCC" if "LFCC" in feature_choice else "MFCC"

model, device = load_trained_model(selected_feature)

if model is None:
  st.warning(
      f"⚠️ ไม่พบไฟล์โมเดลสำหรับ `{selected_feature}` กรุณารันไฟล์ `train.py`"
      f" ที่ตั้งค่า FEATURE_TYPE = '{selected_feature}' ก่อนใช้งาน"
  )
else:
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
      st.sidebar.warning("ไม่พบไฟล์เสียงในโฟลเดอร์ dataset")
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
    st.subheader(
        f"🎧 ไฟล์ที่กำลังตรวจสอบ: `{os.path.basename(target_file)}`"
        f" [โหมด: {selected_feature}]"
    )
    st.audio(target_file)

    try:
      y, sr = librosa.load(target_file, sr=SAMPLE_RATE, duration=4.0)

      feat_db = extract_features(
          y, sr, feature_type=selected_feature, max_len=MAX_LEN
      )

      input_tensor = (
          torch.tensor(feat_db, dtype=torch.float32)
          .unsqueeze(0)
          .unsqueeze(0)
          .to(device)
      )

      with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        pred_label = torch.argmax(probabilities).item()
        confidence = probabilities[pred_label].item() * 100

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
      st.subheader(
          f"📊 การวิเคราะห์สัญญาณและสเปกตรัม ({selected_feature} Spectrogram)"
      )
      fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))

      # 1. Waveform
      librosa.display.waveshow(y, sr=sr, ax=ax[0], color="royalblue")
      ax[0].set(title="Waveform (Time-Domain)")
      ax[0].set_xlabel("Time (s)")
      ax[0].set_ylabel("Amplitude")

      # 2. Spectrogram (MFCC / LFCC)
      cmap_type = "coolwarm" if selected_feature == "MFCC" else "magma"
      img = librosa.display.specshow(
          feat_db, sr=sr, hop_length=512, x_axis="time", ax=ax[1], cmap=cmap_type
      )
      ax[1].set(title=f"{selected_feature} Spectrogram Representation")
      fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

      plt.tight_layout()
      st.pyplot(fig)

    except Exception as e:
      st.error(f"⚠️ เกิดข้อผิดพลาดในการประมวลผลไฟล์เสียง: {e}")
  else:
    st.warning("⚠️ กรุณาเลือกไฟล์ตัวอย่างหรืออัปโหลดไฟล์เสียงเพื่อเริ่มการตรวจสอบ")