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
    return -1
  return times / count

def print_metrics(y_true_list, y_pred_list):
  print(confusion_matrix(*any_transform(y_true_list, y_pred_list)))
  print(f'Accuracy:     {round(accuracy(y_true_list, y_pred_list), 3)}\tWie oft lag der Algorithmus richtig?')
  print(f'Recall (Sns): {round(recall(y_true_list, y_pred_list), 3)}\tWie gut wurden echte Lecks erkannt?')
  print(f'Specificity:  {round(specificity(y_true_list, y_pred_list), 3)}\tWie gut wurde \'alles ok\' erkannt?')
  print(f'Precision:    {round(precision(y_true_list, y_pred_list), 3)}\tWie viele erkannte lecks waren auch wirklich Lecks?')
  print(f'Mean Detection Time: {round(detection_time(y_true_list, y_pred_list), 3)}h\tWie viele Stunden dauerte es bis zum erkennen?')
