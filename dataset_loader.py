import librosa
import numpy as np
import torchaudio
import torchaudio.transforms as T

# 1. โหลดไฟล์เสียงดิบ (Raw Audio)
# สมมติว่าไฟล์เสียงชื่อ 'sample_audio.wav' อัตราการสุ่มตัวอย่าง 16,000 Hz
audio_path = 'sample_audio.wav'
y, sr = librosa.load(audio_path, sr=16000)

print(f"Loaded audio shape: {y.shape}, Sample Rate: {sr}")

# -------------------------------------------------------------
# 2. การสกัดฟีเจอร์แบบที่ 1: MFCC (Mel-Frequency Cepstral Coefficients)
# เน้นความละเอียดด้านความถี่ต่ำ (จำลองการได้ยินของมนุษย์)
# -------------------------------------------------------------
n_mfcc = 20
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
print(f"MFCCs shape: {mfccs.shape}")  # จะได้ขนาด (n_mfcc, จำนวนเฟรมเวลา)

# -------------------------------------------------------------
# 3. การสกัดฟีเจอร์แบบที่ 2: LFCC (Linear-Frequency Cepstral Coefficients)
# เน้นความละเอียดสเกลเชิงเส้น (รักษาความถี่สูงที่ Deepfake มักทิ้งร่องรอยไว้)
# -------------------------------------------------------------
# แปลง numpy array เป็น PyTorch Tensor เพื่อใช้ฟังก์ชันจาก torchaudio
waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # shape: (1, time)

lfcc_transform = T.LFCC(
    sample_rate=sr,
    n_filter=128,  # จำนวนฟิลเตอร์แบบเชิงเส้น
    n_lfcc=20,  # จำนวนค่าสัมประสิทธิ์ที่ต้องการดึงออกมา
    speckwargs={"n_fft": 2048, "hop_length": 512, "win_length": 2048},
)

lfccs = lfcc_transform(waveform)
print(f"LFCCs shape: {lfccs.shape}")  # จะได้ขนาด (1, n_lfcc, จำนวนเฟรมเวลา)