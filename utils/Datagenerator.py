import numpy as np
import pandas as pd
from .Dataloader import Dataloader
from tqdm import tqdm
from utils.helper import shuffle_data
pd.options.mode.chained_assignment = None

class Datagenerator:
  """
  Class used to generate training and testing data.
  Also loads LeakDB dataset.
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

  def gen_dataset(self, size=50, leakage_nodes=['12'], leak_perc=0.5, days_per_sim=5, include_time=True, noise_strength=0.3, numpy=False, shuffle=False, return_nodes=False):
    """Generate a whole dataset containing leakage and non leakage scenarios.

    Args:
      size (int): The numbers of simulations.
      nodes (list of nodes): The nodes that should be used for leakages.
      leak_perc (float [0..1]): Percentage of simulations with leakage.
      days_per_sim (int): The numbers of days per simulation.
      include_time (bool): If 'hour of the day' should be included.
      noise_strength (float): Strength of the noise added.
      numpy (bool): If the data should be converted to numpy arrays.
      shuffle (bool): If the data should be shuffled.
      return_nodes (bool): If the node chosen for the leak should be returned.
    
    Returns:
      X: List of Pandas Dataframes containing node pressures with hour as index.
      y: List of Numpy arrays containing the labels.
    """

    # Calculate amount of leakage scenarios
    size_leak = int(size*leak_perc)

    # Generate simulations
    X, y, y_nodes = [], [], []
    for leak, iters in zip([True, False], [size_leak, size - size_leak]):
      if leak:
        print(f'Generating {iters} leakage scenarios...')
      else:
        print(f'Generating {iters} non leakage scenarios...')
      for i in tqdm(range(iters)):
        if leak:
          node = np.random.choice(leakage_nodes)
        else:
          node = ''
        X_single, y_single = self.gen_single_data(node, num_hours=24*days_per_sim, include_time=include_time, noise_strength=noise_strength)
        X.append(X_single)
        y.append(y_single)
        y_nodes.append(node)
    
    if numpy:
      X, y, y_nodes = np.array(X), np.array(y), np.array(y_nodes)
    
    if shuffle:
      X, y, y_nodes = shuffle_data(X, y, y_nodes)
    
    if return_nodes:
      return X, y, y_nodes
    
    return X, y

  def get_scenario(self, root, scenario, ts_in_h=0.5, include_time=True):
    """Load one scenario of the LeakDB dataset.

    Args:
      root (str): Root path of the dataset.
      scenario (int): The scenario number of the dataset.
      ts_in_h (float): Time step in hours (0.5 = 30min).
      include_time (bool): If 'hour of the day' should be included.
    
    Returns:
      X: Pandas Dataframe containing node pressures with hour as index.
      y: Numpy array containing the labels.
    """

    nodes = list(self.wdn.important_nodes)
    
    # Load dataset
    data = pd.read_csv(root + f'Scenario-{scenario}/Labels.csv')[['Label']]
    for node in nodes:
      data[node] = pd.read_csv(root + f'Scenario-{scenario}/Pressures/Node_{node}.csv')['Value']

    # Add hour info
    data['hour'] = np.arange(0, len(data) * ts_in_h, ts_in_h)
    data['hour of the day'] = data['hour'] % 24
    data['day'] = data['hour'] // 24
    data = data.set_index('hour')

    start = int(10*24//ts_in_h)

    if include_time:
      nodes.append('hour of the day')
    
    X, y = data[nodes][start:], np.array(data['Label'][start:])
    X.reset_index(drop=True, inplace=True)
    
    return X, y
  
  def get_dataset(self, root, size=500, leak_perc=0.5, day_window=10, ts_in_h=0.5, include_time=True, shuffle=True):
    """Get a selection of scenarios from the LeakDB dataset.

    Args:
      root (str): Root path of the dataset.
      size (int): The numbers of scenarios.
      leak_perc (float [0..1]): Percentage of scenarios with leakage.
      day_window (int): The numbers of days for a scenario.
      ts_in_h (float): Time step in hours (0.5 = 30min).
      include_time (bool): If 'hour of the day' should be included.
      shuffle (bool): If the data should be shuffled.
    
    Returns:
      X: List of Pandas Dataframes containing node pressures with hour as index.
      y: List of Numpy arrays containing the labels.
    """
    ts_window = int(day_window * 24 / ts_in_h)

    # Seperate leakage and non leakage scenarios
    labels = pd.read_csv(root + 'Labels.csv')
    labels = labels.set_index('Scenario')
    sc_leak_idx   = np.array(labels.index[labels['Label'] == 1])
    sc_noleak_idx = np.array(labels.index[labels['Label'] == 0])

    # Calculate amount of leakage scenarios
    size_leak = int(size*leak_perc)
    if size_leak > len(sc_leak_idx):
      raise ValueError('Number of wanted leakage scenarios too large')

    # Select scenarios
    scenarios_leak   = np.random.choice(sc_leak_idx, size_leak, replace=False)
    scenarios_noleak = np.random.choice(sc_noleak_idx, size - size_leak, replace=False)

    # Load and crop scenarios
    X, y = [], []
    for leak, scns in zip([True, False], [scenarios_leak, scenarios_noleak]):
      if leak:
        print(f'Getting {len(scenarios_leak)} leakage scenarios...')
      else:
        print(f'Getting {len(scenarios_noleak)} non leakage scenarios...')
      for scn in tqdm(scns):
        X_single, y_single = self.get_scenario(root, scn, ts_in_h, include_time)
        if leak:
          try:
            time_point = np.where(y_single == 1)[0][0] - int(ts_window / 2)
            time_point = min(max(time_point, 0), len(X_single) - ts_window) # Keep day window in range
          except:
            print('Warning: Wrong label. Counting scenario as non leakage.')
            time_point = np.random.randint(len(X_single) - ts_window)
        else:
          time_point = np.random.randint(len(X_single) - ts_window)
        X.append(X_single.iloc[time_point : time_point + ts_window].reset_index())
        y.append(y_single[time_point : time_point + ts_window])
    
    if shuffle:
      X, y, _ = shuffle_data(X, y, y)
    
    return X, y

  def get_full_dataset(self, root, ts_in_h=0.5, include_time=True):
    """Load a whole LeakDB dataset containing leakage and non 
    leakage scenarios.

    Args:
      root (str): Root path of the dataset.
      ts_in_h (float): Time step in hours (0.5 = 30min).
      include_time (bool): If 'hour of the day' should be included.
    
    Returns:
      X: List of Pandas Dataframes containing node pressures with hour as index.
      y: List of Numpy arrays containing the labels.
    """

    # Get scenarios
    X, y = [], []
    for scenario in tqdm(range(1, 1001)):
      X_single, y_single = self.get_scenario(root, scenario, ts_in_h, include_time)

      X.append(X_single)
      y.append(y_single)
        
    return X, y
