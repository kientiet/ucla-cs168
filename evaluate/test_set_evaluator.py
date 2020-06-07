import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from collections import Counter

class TestSetEvaluator:
  def __init__(self, valset, threshold = 0.5):
    self.valset = valset
    self.threshold = threshold

  def eval_all(self, y_pred, y_true, y_pred_label = None):
    logs = self.eval_pictures_level(y_pred, y_true, y_pred_label)
    logs.update(self.eval_patients_level(y_pred, y_true, y_pred_label))

    return logs

  def eval_pictures_level(self, y_pred, y_true, y_pred_label = None):
    auc_score = roc_auc_score(y_true.astype(int), y_pred)

    # Get the label here
    if y_pred_label is None:
      y_pred = (y_pred > self.threshold).astype(int)
      logs = {
        "auc_score/picture-level": auc_score,
        "accuracy_score/picture-level": accuracy_score(y_true, y_pred),
        "f1_score/picture-level": f1_score(y_true, y_pred),
        "recall_score/picture-level": recall_score(y_true, y_pred),
        "precision_score/picture-level": precision_score(y_true, y_pred)
      }
    else:
      logs = {
        "auc_score/picture-level": auc_score,
        "accuracy_score/picture-level": accuracy_score(y_true, y_pred_label),
        "f1_score/picture-level": f1_score(y_true, y_pred_label),
        "recall_score/picture-level": recall_score(y_true, y_pred_label),
        "precision_score/picture-level": precision_score(y_true, y_pred_label)
      }

    return logs

  def eval_patients_level(self, y_pred, y_true, y_pred_label = None):
    patients_table = self.valset.get_patient_status()
    # Check if the order is correct
    labels = patients_table[:, 0].astype(int)
    assert np.array_equal(labels, y_true.astype(int))

    patient_true, patient_pred_real, patient_pred_int = np.array([]), np.array([]), np.array([])
    patients = np.unique(patients_table[:, 1])

    for patient_id in patients:
      idx = np.where(patients_table[:, 1] == patient_id)
      patient_true = np.append(patient_true, y_true[idx[0][0]])

      # Get the confidence in prediction
      patient_pred_real = np.append(patient_pred_real, np.mean(y_pred[idx]))

      # Get the majority vote from the prediction
      if y_pred_label is None:
        voters = (y_pred[idx] > 0.5).astype(int)
      else:
        voters = y_pred_label[idx]

      majority = Counter(voters).most_common(1)
      patient_pred_int = np.append(patient_pred_int, int(majority[0][0]))

    auc_score = roc_auc_score(patient_true * 1.0, patient_pred_real)
    patient_pred_int = patient_pred_int.astype(int)

    logs = {
      "auc_score/patient-level": auc_score,
      "accuracy_score/patient-level": accuracy_score(patient_true.astype(int), patient_pred_int),
      "f1_score/patient-level": f1_score(patient_true.astype(int), patient_pred_int),
      "recall_score/patient-level": recall_score(patient_true.astype(int), patient_pred_int),
      "precision_score/patient-level": precision_score(patient_true.astype(int), patient_pred_int)
    }

    # for key, value in logs.items():
    #   if value == 0.0:
    #     breakpoint()

    return logs