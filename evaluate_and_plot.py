import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import torch


def evaluate_model(model, dataloader, device):
  model.eval()
  all_preds = []
  all_labels = []
  all_probs = []

  with torch.no_grad():
    for inputs, labels in dataloader:
      inputs = inputs.to(device)
      outputs = model(inputs)
      probs = torch.softmax(outputs, dim=1)
      _, preds = torch.max(outputs, 1)

      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.numpy())
      all_probs.extend(probs[:, 1].cpu().numpy())

  report = classification_report(
      all_labels, all_preds, target_names=["Real", "Fake"], output_dict=True
  )
  cm = confusion_matrix(all_labels, all_preds)
  auc_score = roc_auc_score(all_labels, all_probs)

  return report, cm, auc_score, all_labels, all_probs


def plot_confusion_matrix(cm):
  plt.figure(figsize=(6, 5))
  sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap="Blues",
      xticklabels=["Real", "Fake"],
      yticklabels=["Real", "Fake"],
  )
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.title("Confusion Matrix - Audio Deepfake Detection")
  plt.tight_layout()
  plt.show()


def plot_roc_curve(y_true, y_probs):
  fpr, tpr, _ = roc_curve(y_true, y_probs)
  auc_score = roc_auc_score(y_true, y_probs)

  plt.figure(figsize=(6, 5))
  plt.plot(
      fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})"
  )
  plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver Operating Characteristic (ROC)")
  plt.legend(loc="lower right")
  plt.tight_layout()
  plt.show()