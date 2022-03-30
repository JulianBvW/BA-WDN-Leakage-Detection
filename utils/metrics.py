from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from utils.helper import any_transform
import numpy as np

def accuracy(y_true_list, y_pred_list):
  return accuracy_score(*any_transform(y_true_list, y_pred_list))

def recall(y_true_list, y_pred_list):
  return recall_score(*any_transform(y_true_list, y_pred_list))

def specificity(y_true_list, y_pred_list):
  return recall_score(*any_transform(y_true_list, y_pred_list), pos_label=0)

def precision(y_true_list, y_pred_list):
  return precision_score(*any_transform(y_true_list, y_pred_list))

def detection_time(y_true_list, y_pred_list):
  times = 0
  count = 0
  for y_true, y_pred in zip(y_true_list, y_pred_list):
    if 1 in y_true:
      idx = np.where(y_true == 1)[0][0]
      if 1 in y_pred[idx:]:
        times += np.where(y_pred[idx:] == 1)[0][0]
        count += 1
  if count == 0:
    return -float('inf')
  return -times / count

def detection_time_list(y_true_list, y_pred_list):
  times = []
  for y_true, y_pred in zip(y_true_list, y_pred_list):
    if 1 in y_true:
      idx = np.where(y_true == 1)[0][0]
      if 1 in y_pred[idx:]:
        times.append(np.where(y_pred[idx:] == 1)[0][0])
  return times

def detection_time_mean(y_true_list, y_pred_list):
  return np.mean(detection_time_list(y_true_list, y_pred_list))

def detection_time_std(y_true_list, y_pred_list):
  return np.std(detection_time_list(y_true_list, y_pred_list))

def detection_time_median(y_true_list, y_pred_list):
  return np.median(detection_time_list(y_true_list, y_pred_list))

def print_metrics(y_true_list, y_pred_list):
  print(confusion_matrix(*any_transform(y_true_list, y_pred_list)))
  print(f'Accuracy:     {round(accuracy(y_true_list, y_pred_list), 3)}\tWie oft lag der Algorithmus richtig?')
  print(f'Recall (Sns): {round(recall(y_true_list, y_pred_list), 3)}\tWie gut wurden echte Lecks erkannt?')
  print(f'Specificity:  {round(specificity(y_true_list, y_pred_list), 3)}\tWie gut wurde \'alles ok\' erkannt?')
  print(f'Precision:    {round(precision(y_true_list, y_pred_list), 3)}\tWie viele erkannte lecks waren auch wirklich Lecks?')
  print(f'Detection Time      \tWie viele Zeiteinheiten dauerte es bis zum erkennen?')
  print(f' -> Mean: {round(detection_time_mean(y_true_list, y_pred_list), 3)}ts\tStd: {round(detection_time_std(y_true_list, y_pred_list), 3)}ts\tMedian: {round(detection_time_median(y_true_list, y_pred_list), 3)}ts')
