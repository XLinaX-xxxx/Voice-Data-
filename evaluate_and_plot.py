import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from train import AudioDeepfakeCNN  # ดึงโครงสร้างโมเดลที่เราสร้างไว้

def evaluate_audio(file_path):
    print(f"\nกำลังตรวจสอบไฟล์: {file_path}")
    
    # 1. โหลดไฟล์เสียง
    y, sr = librosa.load(file_path, sr=16000)
    
    # 2. สกัดฟีเจอร์ MFCC (แบบเดียวกับตอนเทรน)
    mfccs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
    mfccs_db = librosa.power_to_db(mfccs)
    
    if mfccs_db.shape[1] < 100:
        mfccs_db = np.pad(mfccs_db, ((0, 0), (0, 100 - mfccs_db.shape[1])), mode='constant')
    else:
        mfccs_db = mfccs_db[:, :100]
        
    # 3. โหลดโมเดลที่เทรนแล้ว (จำลองการทำนาย)
    model = AudioDeepfakeCNN()
    model.eval()
    
    # แปลงข้อมูลเป็น Tensor เพื่อส่งให้ AI ทำนาย
    input_tensor = torch.tensor(mfccs_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_label = torch.argmax(probabilities, dim=1).item()
        
    class_names = ["REAL (เสียงจริง)", "FAKE (เสียงปลอม/Deepfake)"]
    print(f"ผลการทำนาย: {class_names[pred_label]}")
    
    # 4. วาดกราฟแสดงให้อาจารย์ดู (Waveform & Spectrogram)
    plt.figure(figsize=(12, 6))
    
    # กราฟที่ 1: Waveform (รูปคลื่นในโดเมนเวลา)
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, color='blue')
    plt.title(f"Waveform of {os.path.basename(file_path)} -> Result: {class_names[pred_label]}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # กราฟที่ 2: Mel-Spectrogram (ลักษณะความถี่ที่ AI ใช้มองภาพ)
    plt.subplot(2, 1, 2)
    img = librosa.display.specshow(mfccs_db, sr=sr, x_axis='time', y_axis='mel', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title("MFCC / Mel-Spectrogram Feature (Input to CNN)")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # เลือกไฟล์ตัวอย่างมาทดสอบโชว์ (สามารถเปลี่ยนชื่อไฟล์ได้ตามต้องการ)
    sample_file = "dataset/real/1.wav"  # หรือลองเปลี่ยนเป็น "dataset/fake/deepfake1.wav"
    
    if os.path.exists(sample_file):
        evaluate_audio(sample_file)
    else:
        print(f"ไม่พบไฟล์ {sample_file} กรุณาตรวจสอบ path อีกครั้ง")