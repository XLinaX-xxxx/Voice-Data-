from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def calculate_eer(y_true, y_scores):
  """คำนวณค่า Equal Error Rate (EER)

  y_true: ป้ายกำกับจริง (0 = เสียงจริง/Bonafide, 1 = เสียงปลอม/Spoof)
  y_scores: ค่าความน่าจะเป็นที่โมเดลทำนายว่าเป็น Spoof
  """
  # คำนวณ False Positive Rate (FPR) และ True Positive Rate (TPR)
  fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

  # หาจุดที่ FPR ตัดกับ (1 - TPR) ซึ่งก็คือจุดที่ FAR = FRR (EER)
  eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
  return eer, fpr, tpr


# --- ตัวอย่างการเรียกใช้งาน ---
# สมมติเราทดสอบด้วยชุดข้อมูล Test Set
# y_true_test = [0, 1, 0, 1, 0, ...]
# y_scores_mfcc = model_mfcc_predict(...)
# y_scores_lfcc = model_lfcc_predict(...)

# eer_mfcc, _, _ = calculate_eer(y_true_test, y_scores_mfcc)
# eer_lfcc, _, _ = calculate_eer(y_true_test, y_scores_lfcc)

# print(f"MFCC EER: {eer_mfcc * 100:.2f}%")
# print(f"LFCC EER: {eer_lfcc * 100:.2f}%")