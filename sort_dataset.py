import os
import shutil

# --- ตั้งค่า Path ตามจริงในเครื่องของคุณ ---
AUDIO_SOURCE_DIR = r"D:\Voice data\archive\LA\LA\ASVspoof2019_LA_train\flac"
PROTOCOL_FILE = r"D:\Voice data\archive\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)


def organize_asvspoof_dataset():
  if not os.path.exists(PROTOCOL_FILE):
    print(f"❌ ไม่พบไฟล์ Protocol ที่: {PROTOCOL_FILE}")
    return

  if not os.path.exists(AUDIO_SOURCE_DIR):
    print(f"❌ ไม่พบโฟลเดอร์เก็บไฟล์เสียงที่: {AUDIO_SOURCE_DIR}")
    return

  print("--- เริ่มกระบวนการอ่านไฟล์ Protocol และคัดแยกเสียง ---")
  copied_real = 0
  copied_fake = 0
  not_found = 0
  total_lines = 0

  with open(PROTOCOL_FILE, "r") as f:
    for line in f:
      total_lines += 1
      parts = line.strip().split()
      if len(parts) >= 4:
        # โครงสร้าง Protocol ของ ASVspoof 2019: [speaker_id, file_name, system_id, key]
        # เช่น [LA_0079, LA_T_1138215, -, -, bonafide] -> parts[1] คือชื่อไฟล์, parts[-1] คือ label
        file_name = parts[1]
        label = parts[-1]  # ใช้ตำแหน่งสุดท้ายเพื่อความปลอดภัย ('bonafide' หรือ 'spoof')

        src_file = os.path.join(AUDIO_SOURCE_DIR, f"{file_name}.flac")

        if os.path.exists(src_file):
          if label == "bonafide":
            shutil.copy(src_file, os.path.join(REAL_DIR, f"{file_name}.flac"))
            copied_real += 1
          elif label == "spoof":
            shutil.copy(src_file, os.path.join(FAKE_DIR, f"{file_name}.flac"))
            copied_fake += 1
        else:
          not_found += 1

  print("\n--- คัดแยกไฟล์เสร็จเรียบร้อยแล้ว! ---")
  print(f"📄 อ่านไฟล์ Protocol ทั้งหมด: {total_lines:,} บรรทัด")
  print(f"📁 เสียงมนุษย์จริง (Real / bonafide): {copied_real:,} ไฟล์")
  print(f"📁 เสียงสังเคราะห์ (Fake / spoof): {copied_fake:,} ไฟล์")
  if not_found > 0:
    print(f"⚠️ หาไฟล์ต้นฉบับไม่พบ: {not_found:,} ไฟล์")


if __name__ == "__main__":
  organize_asvspoof_dataset()