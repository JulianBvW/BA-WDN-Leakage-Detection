# General
import sys
import numpy as np
import pandas as pd

# SKLearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Own
from utils.Network import WDN
from utils.Datagenerator import Datagenerator
from utils.metrics import accuracy, recall, specificity, precision, detection_time_mean, detection_time_std, detection_time_median
from models.Classification import ClassificationModel
from models.RegressionEnsamble import RegressionEnsamble

####### Main
LEAKDB_PATH = '../Net1_CMH/'

scoring = {'accuracy': make_scorer(accuracy),
           'recall': make_scorer(recall),
           'specificity': make_scorer(specificity),
           'precision': make_scorer(precision),
           'detection_time_mean': make_scorer(detection_time_mean),
           'detection_time_std': make_scorer(detection_time_std),
           'detection_time_median': make_scorer(detection_time_median)}

# New class for Neural Networks

class NNSize(object):
    def __init__(self, shape_min=1, shape_max=3, size_min=5, size_max=24):
        self.shape_min = shape_min
        self.shape_max = shape_max
        self.size_min = size_min
        self.size_max = size_max

    def rvs(self, random_state=None):
        #np.random.seed(random_state)
        shape = np.random.randint(self.shape_min, self.shape_max+1)
        return tuple([np.random.randint(self.size_min, self.size_max+1) for _ in range(shape)])

# GridSearch function

def do_gridsearch(name, X, y, model, parameters):
  print(f'# {name}...')

  grid = GridSearchCV(model, parameters, scoring=scoring, refit='accuracy', cv=5, verbose=2)
  grid.fit(X, y)

  pd.DataFrame(grid.cv_results_).to_csv(f'results/{name}.csv')

# Generate classification results

def rgen_classification(X, y):
  general_params = {'medfilt_kernel_size': [3, 5, 7, 9, 11]}
  print('### GENERATING CLASSIFICATION RESULTS...')

  model_knn = ClassificationModel(KNeighborsClassifier())
  parameters_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
  do_gridsearch('adhoc_classification_knn', X, y, model_knn, dict(general_params, **parameters_knn))

  """
  model_svc = ClassificationModel(SVC())
  parameters_svc = {'kernel': ['linear', 'poly', 'rbf'], 
                    'C': [10**e for e in range(-1, 2)], 
                    'gamma': [0.001, 0.002, 0.003]}
  do_gridsearch('classification_svc', X, y, model_svc, dict(general_params, **parameters_svc))"""

  model_mlp = ClassificationModel(MLPClassifier())
  parameters_mlp = {'hidden_layer_sizes': [NNSize().rvs() for _ in range(30)], 
                    'learning_rate': ['constant', 'adaptive'], 
                    'activation': ['logistic', 'tanh', 'relu']}
  do_gridsearch('adhoc_classification_mlp', X, y, model_mlp, dict(general_params, **parameters_mlp))

  print('### RESULTS GENERATED...')

# Generate regression results

def rgen_regression(X, y, model='all'):
  general_params = {'medfilt_kernel_size': [3, 5, 7, 9, 11], 'th_mode': ['simple', 'daytime'], 'th_multiplier': [1 + 0.2*i for i in range(5)], 'th_majority': [i / 10 for i in range(1, 11)]}
  print('### GENERATING REGRESSION RESULTS...')

  if model in ['all', 'lr']:
    rgen_regression_lr(X, y, general_params)
  if model in ['all', 'ridge']:
    rgen_regression_ridge(X, y, general_params)
  if model in ['all', 'lasso']:
    rgen_regression_lasso(X, y, general_params)
  #if model in ['all', 'dt']:
    #rgen_regression_dt(X, y, general_params)
  #if model in ['all', 'rf']:
    #rgen_regression_rf(X, y, general_params)
  if model in ['all', 'knn']:
    rgen_regression_knn(X, y, general_params)
  #if model in ['all', 'svr']:
    #rgen_regression_svr(X, y, general_params)
  if model in ['all', 'mlp']:
    rgen_regression_mlp(X, y, general_params)

  print('### RESULTS GENERATED...')

# Helper

def rgen_regression_lr(X, y, general_params):
  model_lr = RegressionEnsamble(LinearRegression())
  parameters_lr = {}
  do_gridsearch('adhoc_regression_lr', X, y, model_lr, dict(general_params, **parameters_lr))

def rgen_regression_ridge(X, y, general_params):
  model_ridge = RegressionEnsamble(Ridge())
  parameters_ridge = {'alpha': [0.333*i for i in range(1, 7)]}
  do_gridsearch('adhoc_regression_ridge', X, y, model_ridge, dict(general_params, **parameters_ridge))

def rgen_regression_lasso(X, y, general_params):
  model_lasso = RegressionEnsamble(Lasso())
  parameters_lasso = {'alpha': [0.333*i for i in range(1, 7)], 'max_iter': [2000]}
  do_gridsearch('adhoc_regression_lasso', X, y, model_lasso, dict(general_params, **parameters_lasso))

def rgen_regression_dt(X, y, general_params):
  model_dt = RegressionEnsamble(DecisionTreeRegressor())
  parameters_dt = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                   'splitter': ['best', 'random']}
  do_gridsearch('regression_dt', X, y, model_dt, dict(general_params, **parameters_dt))

def rgen_regression_rf(X, y, general_params):
  model_rf = RegressionEnsamble(RandomForestRegressor())
  parameters_rf = {'criterion': ['squared_error', 'absolute_error']}
  do_gridsearch('regression_rf', X, y, model_rf, dict(general_params, **parameters_rf))

def rgen_regression_knn(X, y, general_params):
  model_knn = RegressionEnsamble(KNeighborsRegressor())
  parameters_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
  do_gridsearch('adhoc_regression_knn', X, y, model_knn, dict(general_params, **parameters_knn))

def rgen_regression_svr(X, y, general_params):
  model_svr = RegressionEnsamble(SVR())
  parameters_svr = {'kernel': ['linear', 'poly', 'rbf'], 
                    'C': [10**e for e in range(-1, 2)], 
                    'gamma': [0.001, 0.002, 0.003]}
  do_gridsearch('regression_svr', X, y, model_svr, dict(general_params, **parameters_svr))

def rgen_regression_mlp(X, y, general_params):
  model_mlp = RegressionEnsamble(MLPRegressor())
  parameters_mlp = {'hidden_layer_sizes': [NNSize().rvs() for _ in range(30)], 
                    'learning_rate': ['constant', 'adaptive'], 
                    'activation': ['logistic', 'tanh', 'relu']}
  do_gridsearch('adhoc_regression_mlp', X, y, model_mlp, dict(general_params, **parameters_mlp))

# Main

def main():
  wdn = WDN("nets/Net1.inp", ['10', '11','12','13','21','22','23','31','32'])
  gen = Datagenerator(wdn)

  print('### LOADING DATASET...')
  #X, y = gen.get_dataset(LEAKDB_PATH, size=500)
  X, y = gen.gen_dataset(size=150*2, leakage_nodes=wdn.important_nodes, shuffle=True, return_nodes=False)

  if sys.argv[1] == 'classification':
    rgen_classification(X, y)
  if sys.argv[1] == 'regression':
    if len(sys.argv) > 2:
      rgen_regression(X, y, model=sys.argv[2])
    else:
      rgen_regression(X, y)

if __name__ == '__main__':
  main()
