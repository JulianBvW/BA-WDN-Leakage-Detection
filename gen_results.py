# General
import sys
import pandas as pd

# SKLearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

from sklearn.neighbors import KNeighborsClassifier

# Own
from utils.Network import WDN
from utils.Datagenerator import Datagenerator
from utils.metrics import accuracy, recall, specificity, precision, detection_time_mean, detection_time_std, detection_time_median
from models.Classification import ClassificationModel

####### Main
LEAKDB_PATH = '../Net1_CMH/'

scoring = {'accuracy': make_scorer(accuracy),
           'recall': make_scorer(recall),
           'specificity': make_scorer(specificity),
           'precision': make_scorer(precision),
           'detection_time_mean': make_scorer(detection_time_mean),
           'detection_time_std': make_scorer(detection_time_std),
           'detection_time_median': make_scorer(detection_time_median)}

# Generate classification results
def rgen_classification(X, y):
  print('STARTING GENERATION OF CLASSIFICATION RESULTS...')

  parameters_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
  model_knn = ClassificationModel(KNeighborsClassifier())
  
  grid = GridSearchCV(model_knn, parameters_knn, scoring=scoring, cv=5, verbose=1)
  grid.fit(X, y)

  pd.DataFrame(grid.cv_results_).to_csv('results/bla.csv')

def main():
  wdn = WDN("nets/Net1.inp", ['10', '11','12','13','21','22','23','31','32'])
  gen = Datagenerator(wdn)
  X, y = gen.get_dataset(LEAKDB_PATH)

  if sys.argv[1] == 'classification':
    rgen_classification(X, y)

if __name__ == '__main__':
  main()