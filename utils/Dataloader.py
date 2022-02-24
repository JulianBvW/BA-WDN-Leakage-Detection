import numpy as np
import pandas as pd

class Dataloader:
  """
  Class used to reshape WDN data.
  """

  def __init__(self, simulation_results, important_nodes):
    """Create a dataset based on the simulated results.

    Args:
      simulation_results (Dataframe): Simulated results including hour, time of day, day nad pressures.
      importnat_nodes (iterable of nodes): List of nodes (Strings) interesting for ML and plotting.
    """
    self.data = simulation_results
    self.important_nodes = important_nodes
  
  def get_nodes(self, nodes=[], include_time=False, include_day=False):
    """Select specific nodes or just the important nodes.

    Args:
      nodes (iterable of nodes) (optional): The nodes you want to get.
      include_time (Bool) (optional): Set flag to include time of day and day columns.
    
    Returns:
      Pandas Dataframe containing node pressures with hour as index.
    """
    nodelist = []

    nodelist += nodes

    if not nodes:
      nodelist += self.important_nodes
    
    if include_time:
      nodelist.append('hour of the day')
    
    if include_day:
      nodelist.append('day')
    
    print(nodelist)
      
    return self.data.loc[:, nodelist].copy()
  
  def get_days_at_hour(self, hour, nodes=[]):
    """Set day as index while looking at a specific hour.

    Args:
      hour (Int): The hour of the day.
      nodes (iterable of nodes) (optional): The nodes you want to get.
    
    Returns:
      pandas Dataframe containing pressure values for the specified hour.
    """
    at_hour = self.data[self.data['hour of the day'] == hour].copy().set_index('day')
    
    if nodes:
      return at_hour.loc[:, nodes]

    return at_hour.loc[:, self.important_nodes]
  
  def get_days_at_hours(self, node, hours=[]):
    """Set day as index while looking at (specific) hours for a specific node.

    Args:
      node (node): The node you want to get.
      hours (iterable of int) (optional): Specific hours.
    
    Returns:
      pandas Dataframe containing pressures for one node at different hours.
    """
    if not hours:
      hours = range(24)

    data = self.get_days_at_hour(23, [node]).loc[:, []]
    for i in hours:
      data[i] = self.get_days_at_hour(i, [node])[node].values.tolist()[:len(data)]
    return data
