from torch.utils.data import DataLoader, Dataset


class AudioDataset(Dataset):

  def __init__(self, file_list, labels):
    self.file_list = file_list
    self.labels = labels

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, idx):
    # โหลดไฟล์เสียงทีละไฟล์ตาม index เพื่อประหยัด RAM
    audio_path = self.file_list[idx]
    y, sr = librosa.load(audio_path, sr=16000)

    # สกัดฟีเจอร์ตรงนี้ (เลือกได้ว่าจะทำ MFCC หรือ LFCC)
    feature = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # แปลงเป็น Tensor แล้วส่งออกไป
    return torch.tensor(feature).unsqueeze(0), torch.tensor(
        self.labels[idx]
    )


# เรียกใช้งานผ่าน DataLoader เพื่อแบ่งส่งข้อมูลทีละ Batch
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)