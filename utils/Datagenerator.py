import numpy as np
import pandas as pd
from .Dataloader import Dataloader
from tqdm import tqdm
pd.options.mode.chained_assignment = None

class Datagenerator:
  """
  Class used to generate training and testing data.
  """

  def __init__(self, wdn):
    """Create the generator.

    Args:
      wdn (WDN): The network class.
    """
    self.wdn = wdn
  
  def gen_single_data(self, leakage_node='', num_hours=30*24, include_time=False, noise_strength=0.3):
    """Generate a time series of data possibly containing a leakage from the
       last half until the end.

    Args:
      leakage_node (node or ''): The node for the leakage. Empty for no leak.
      num_hours (int): The numbers of hours to simulate.
      include_time (bool): If 'hour of the day' should be included.
      noise_strength (float): Strength of the noise added.
    
    Returns:
      X: Pandas Dataframe containing node pressures with hour as index.
      y: Numpy array containing the labels.
    """
  
    # Labels
    y = np.zeros((num_hours+1,))
  
    # Construct leakages
    leakages = []
    if leakage_node:
      leakage_start = np.random.randint(num_hours // 2, num_hours * 9 // 10)
      leakage_strength = round(0.0009 + np.random.rand() / 2000, 4)
      leakages.append((leakage_node, leakage_strength, leakage_start+10*24, num_hours+10*24))
      y[leakage_start:] = 1
    
    # Run simulation and remove first 10 convergence days
    p = self.wdn.simulate(num_hours+10*24, leakages)[10*24:]
  
    # Add noise
    noise = np.random.normal(0, noise_strength, [len(p), len(self.wdn.important_nodes)])
    p.loc[:, self.wdn.important_nodes] = p.loc[:, self.wdn.important_nodes] + noise
  
    # Features
    data = Dataloader(p, self.wdn.important_nodes)
    X = data.get_nodes(include_time=include_time)
    X.reset_index(drop=True, inplace=True)
  
    return X, y

  def gen_multi_data(self, leakage_node='', num_sim=5, days_per_sim=10, include_time=False, noise_strength=0.15):
    """Generate multiple time series of data possibly containing leakages from
      the last half until the end of each series.

    Args:
      leakage_node (node or ''): The node for the leakage. Empty for no leak.
      num_sim (int): The numbers of simulations.
      days_per_sim (int): The numbers of days per simulation.
      include_time (bool): If 'hour of the day' should be included.
      noise_strength (float): Strength of the noise added.
    
    Returns:
      X: Pandas Dataframe containing node pressures with hour as index.
      y: Numpy array containing the labels.
    """

    # Generate simulation results
    X_list = []
    y_list = []
    for i in range(num_sim):
      X, y = self.gen_single_data(leakage_node, num_hours=24*days_per_sim, include_time=include_time, noise_strength=noise_strength)
      X_list.append(X)
      y_list.append(y)
  
    # Concatinate the results
    X_concat = pd.concat(X_list)
    y_concat = np.concatenate(y_list)
    X_concat.reset_index(drop=True, inplace=True)

    return X_concat, y_concat


  def gen_dataset(self, size=50, leak_perc=0.5, days_per_sim=5, include_time=True, noise_strength=0.3):
    """Generate a whole dataset containing leakage and non leakage scenarios.

    Args:
      size (int): The numbers of simulations.
      leak_perc (float [0..1]): Percentage of simulations with leakage.
      days_per_sim (int): The numbers of days per simulation.
      include_time (bool): If 'hour of the day' should be included.
      noise_strength (float): Strength of the noise added.
    
    Returns:
      X: List of Pandas Dataframes containing node pressures with hour as index.
      y: List of Numpy arrays containing the labels.
    """

    # Calculate amount of leakage scenarios
    size_leak = int(size*leak_perc)

    # Generate simulations
    X, y = [], []
    for node, iters in zip(['12', ''], [size_leak, size - size_leak]):
      if node == '':
        print(f'Generating {iters} non leakage scenarios...')
      else:
        print(f'Generating {iters} leakage scenarios...')
      for i in tqdm(range(iters)):
        X_single, y_single = self.gen_single_data(node, num_hours=24*days_per_sim, include_time=include_time, noise_strength=noise_strength)
        X.append(X_single)
        y.append(y_single)
    
    return X, y
