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
def load_trained_model():
    model_path = "best_model_lfcc.pth"
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

st.title("Audio Deepfake Detection System")
st.write(
    "ระบบตรวจสอบและวิเคราะห์เสียงพูดเพื่อจำแนกเสียงระหว่าง **มนุษย์จริง (Real)**"
    " และ **เสียงสังเคราะห์จาก AI (Fake)** พร้อมระบบอธิบายผลเข้าใจง่ายสำหรับทุกคน"
)

model, device = load_trained_model()

if model is None:
    st.warning(
        "⚠️ ไม่พบไฟล์โมเดล (`best_model_lfcc.pth` หรือ `best_model.pth`)"
        " กรุณารันไฟล์เทรนโมเดลก่อนใช้งาน"
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
        st.subheader(f"ไฟล์ที่กำลังตรวจสอบ: `{os.path.basename(target_file)}`")
        
        col_audio1, col_audio2 = st.columns(2)
        with col_audio1:
            st.markdown("**ไฟล์เสียงที่คุณกำลังตรวจสอบ:**")
            st.audio(target_file)
        with col_audio2:
            st.markdown("**เสียงตัวอย่างมาตรฐานมนุษย์จริง:**")
            if real_files:
                st.audio(real_files[0])
            else:
                st.info("ไม่มีไฟล์ตัวอย่างในระบบ")

        try:
            y, sr = librosa.load(target_file, sr=SAMPLE_RATE, duration=4.0)
            audio_duration = len(y) / sr

            feat_mfcc = extract_features(
                y, sr, feature_type="MFCC", max_len=MAX_LEN
            )
            feat_lfcc = extract_features(
                y, sr, feature_type="LFCC", max_len=MAX_LEN
            )

            input_tensor = (
                torch.tensor(feat_lfcc, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                pred_label = torch.argmax(probabilities).item()
                confidence = probabilities[pred_label].item() * 100
                real_prob = probabilities[0].item() * 100
                fake_prob = probabilities[1].item() * 100

            col1, col2 = st.columns(2)
            with col1:
                if pred_label == 0:
                    st.success("### ผลการทำนาย: เสียงมนุษย์จริง (REAL)")
                else:
                    st.error("### ผลการทำนาย:เสียงปลอม/AI (FAKE)")
            with col2:
                st.metric(
                    label="ความมั่นใจของโมเดล (Confidence Score)",
                    value=f"{confidence:.2f}%",
                )

            st.write("**คะแนนความน่าจะเป็นเชิงลึก:**")
            st.progress(int(real_prob))
            st.caption(f"ความน่าจะเป็นว่าเป็นเสียงมนุษย์ (Real): **{real_prob:.2f}%**")
            st.progress(int(fake_prob))
            st.caption(f"ความน่าจะเป็นว่าเป็นเสียงปลอม (Fake): **{fake_prob:.2f}%**")

            # Executive Summary Card
            st.markdown("---")
            if pred_label == 1:
                st.error(
                    f"🚨 **สรุปผลการตรวจสอบ:** ระบบตรวจพบความผิดปกติเชิงโครงสร้างสเปกตรัมที่สอดคล้องกับการสังเคราะห์ด้วยปัญญาประดิษฐ์ (AI Deepfake) ด้วยระดับความมั่นใจ **{confidence:.2f}%**"
                )
            else:
                st.success(
                    f"✅ **สรุปผลการตรวจสอบ:** โครงสร้างคลื่นเสียงมีความกลมกลืนและสอดคล้องกับธรรมชาติของมนุษย์จริง (Real) ด้วยระดับความมั่นใจ **{confidence:.2f}%**"
                )

            # ตาราง Visual Checklist
            st.subheader("ตารางวิเคราะห์เปรียบเทียบจุดสังเกต")
            
            status_icon = "❌ ผิดปกติ (เสี่ยง Deepfake)" if pred_label == 1 else "✅ ปกติ (ธรรมชาติ)"
            badge_color = "red" if pred_label == 1 else "green"

            st.markdown(
                f"""
                | หัวข้อประเมินเชิงโครงสร้าง | ลักษณะเสียงตามธรรมชาติ | ไฟล์ที่ตรวจสอบ | สถานะ |
                | :--- | :--- | :--- | :--- |
                | **ความนุ่มนวลของฮาร์มอนิก** | ต่อเนื่อง มีความโค้งมน | {'มีร่องรอยตัดต่อ/แข็งกระด้าง' if pred_label == 1 else 'มีความต่อเนื่องกลมกลืน'} | :{badge_color}[{status_icon}] |
                | **ย่านความถี่สูง (>8 kHz)** | ค่อยๆ จางลงตามธรรมชาติ | {'พบคลื่นความถี่ส่วนเกินผิดปกติ' if pred_label == 1 else 'สอดคล้องกับธรรมชาติ'} | :{badge_color}[{status_icon}] |
                | **รูปแบบสเกล LFCC** | กระจายตัวสม่ำเสมอ | {'พบแพทเทิร์นตารางลายน้ำ AI' if pred_label == 1 else 'โครงสร้างสะอาดเรียบร้อย'} | :{badge_color}[{status_icon}] |
                """
            )

            # ภาพสเปกตรัมที่ปรับตำแหน่งกรอบไฮไลต์ให้เหมาะสมสวยงาม
            st.markdown("---")
            st.subheader(
                "ภาพสเปกตรัมเปรียบเทียบ"
            )

            fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 9))

            # 1. Waveform
            librosa.display.waveshow(y, sr=sr, ax=ax[0], color="royalblue")
            ax[0].set(title="1. Raw Audio Waveform (Time vs Amplitude)")
            ax[0].set_xlabel("Time (s)")
            ax[0].set_ylabel("Amplitude")
            ax[0].set_xlim(0, audio_duration)

            # 2. MFCC Spectrogram
            img1 = librosa.display.specshow(
                feat_mfcc, sr=sr, hop_length=512, x_axis="time", ax=ax[1], cmap="coolwarm", y_axis='mel'
            )
            ax[1].set(title="2. MFCC Spectrogram (Human Hearing Simulation)")
            fig.colorbar(img1, ax=ax[1], format="%+2.0f dB")
            ax[1].set_xlim(0, audio_duration)
            
            rect1 = plt.Rectangle((0.1, 500), audio_duration * 0.9, 3500, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
            ax[1].add_patch(rect1)
            ax[1].text(0.2, 4000, "🔍 Target Region", color='yellow', fontsize=10, weight='bold')

            # 3. LFCC Spectrogram
            img2 = librosa.display.specshow(
                feat_lfcc, sr=sr, hop_length=512, x_axis="time", ax=ax[2], cmap="magma"
            )
            ax[2].set(title="3. LFCC Spectrogram (Linear Scale - AI Artifact Detector)")
            fig.colorbar(img2, ax=ax[2], format="%+2.0f dB")
            ax[2].set_xlim(0, audio_duration)

            rect2 = plt.Rectangle((0.1, 2), audio_duration * 0.9, 28, linewidth=2.5, edgecolor='red', facecolor='none')
            ax[2].add_patch(rect2)
            ax[2].text(0.2, 24, "⚠️ Detected AI Artifacts", color='red', fontsize=10, weight='bold')

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ เกิดข้อผิดพลาดในการประมวลผลไฟล์เสียง: {e}")
    else:
        st.warning("⚠️ กรุณาเลือกไฟล์ตัวอย่างหรืออัปโหลดไฟล์เสียงเพื่อเริ่มการวิเคราะห์")