import os
import torch
from torch.utils.data import Dataset
import torchaudio
from audiomentations import Compose, AddGaussianNoise, PitchShift

class ASVspoof2019PADataset(Dataset):
    def __init__(self, protocol_file, audio_dir, max_len=64600, augment=False):
        """
        Args:
            protocol_file (str): พาธไปยังไฟล์ protocol เช่น ASVspoof2019.PA.cm.train.trn.txt
            audio_dir (str): พาธไปยังโฟลเดอร์เก็บไฟล์เสียง .flac เช่น ASVspoof2019_PA_train/
            max_len (int): ความยาวมาตรฐานของตัวอย่างเสียง (Sample points)
            augment (bool): เปิดใช้งาน Data Augmentation เพื่อช่วยให้โมเดลทนต่อเสียงรบกวน
        """
        self.audio_dir = audio_dir
        self.max_len = max_len
        self.data = []
        
        # ระบบ Data Augmentation ช่วยจำลองเสียงรบกวนและสภาพแวดล้อมที่หลากหลาย
        self.augment = augment
        self.augmentation = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5), # เพิ่มเสียงซ่ารบกวน
            PitchShift(min_semitones=-2, max_semitones=2, p=0.3)              # ปรับระดับเสียงสูงต่ำ
        ])

        # อ่านไฟล์ Protocol ตามโครงสร้าง ASVspoof 2019 PA
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    file_name = parts[1]  # ชื่อไฟล์เสียง
                    key = parts[4]        # ป้ายกำกับ: 'bonafide' หรือ 'spoof'
                    
                    # แปลง Label: 0 = เสียงจริง (Real), 1 = เสียงปลอม (Fake)
                    label = 0 if key == 'bonafide' else 1
                    self.data.append((file_name, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name, label = self.data[idx]
        audio_path = os.path.join(self.audio_dir, f"{file_name}.flac")
        
        # โหลดไฟล์เสียงด้วย torchaudio (มาตรฐาน 16kHz)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # ถ้าเป็นเสียงสเตอริโอ (2 ช่องสัญญาณ) ให้ยุบรวมเป็นโมโน
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        audio_np = waveform.squeeze().numpy()

        # ทำ Data Augmentation เฉพาะตอน Train เพื่อแก้ปัญหา Overfitting และ Noise
        if self.augment:
            audio_np = self.augmentation(samples=audio_np, sample_rate=sample_rate)

        # ปรับความยาวเสียงให้เท่ากัน (Padding หรือ Truncating)
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32)
        if audio_tensor.shape[0] > self.max_len:
            audio_tensor = audio_tensor[:self.max_len]
        else:
            padding = self.max_len - audio_tensor.shape[0]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

        # แปลงเป็น Mel-Spectrogram เพื่อป้อนเข้า 2D-CNN
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        mel_spec = mel_transform(audio_tensor.unsqueeze(0))
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        return mel_spec_db, torch.tensor(label, dtype=torch.long)