import os
import random
import shutil

# --- กำหนด Path เฉพาะชุด LA ตามโครงสร้างของคุณ ---
LA_AUDIO_DIR = r"D:\Voice data\archive\LA\LA\ASVspoof2019_LA_train\flac"
LA_PROTOCOL = r"D:\Voice data\archive\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

# โฟลเดอร์ปลายทาง
OUTPUT_DIR = r"dataset"
REAL_DIR = os.path.join(OUTPUT_DIR, "real")
FAKE_DIR = os.path.join(OUTPUT_DIR, "fake")

# ล้างโฟลเดอร์เก่าทิ้งก่อนเพื่อความสะอาด
if os.path.exists(OUTPUT_DIR):
  shutil.rmtree(OUTPUT_DIR)

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)


def main():
  print("--- เริ่มดึงข้อมูลเฉพาะชุด LA (Real/Fake อย่างละ 2580 ไฟล์) ---")

  real_list = []
  fake_list = []

  if not os.path.exists(LA_PROTOCOL):
    print(f"⚠️ ไม่พบไฟล์ Protocol: {LA_PROTOCOL}")
    return

  with open(LA_PROTOCOL, "r", encoding="utf-8") as f:
    for line in f:
      parts = line.strip().split()
      if len(parts) >= 5:
        file_id = parts[1]
        label = parts[4]  # bonafide หรือ spoof

        filename = f"{file_id}.flac"
        source_path = os.path.join(LA_AUDIO_DIR, filename)

        if os.path.exists(source_path):
          if label == "bonafide":
            real_list.append(source_path)
          elif label == "spoof":
            fake_list.append(source_path)

  print(
      f"🔍 พบข้อมูล LA ดั้งเดิม -> Real: {len(real_list)} ไฟล์ | Fake:"
      f" {len(fake_list)} ไฟล์"
  )

  # กำหนดเป้าหมายอย่างละ 2580 ไฟล์
  target_count = 2580
  selected_real = random.sample(real_list, min(target_count, len(real_list)))
  selected_fake = random.sample(fake_list, min(target_count, len(fake_list)))

  print(
      f"✨ สุ่มคัดเลือก -> Real: {len(selected_real)} ไฟล์ | Fake:"
      f" {len(selected_fake)} ไฟล์"
  )
  print("กำลังคัดลอกไฟล์ลงโฟลเดอร์ dataset กรุณาสักครู่...")

  for src in selected_real:
    shutil.copy(src, os.path.join(REAL_DIR, os.path.basename(src)))

  for src in selected_fake:
    shutil.copy(src, os.path.join(FAKE_DIR, os.path.basename(src)))

  print("--- เตรียมชุดข้อมูล LA สำเร็จเรียบร้อย! ---")


if __name__ == "__main__":
  main()