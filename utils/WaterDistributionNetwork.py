import wntr
import numpy as np
import pandas as pd

class WDN:
  """
  Class used to combine common practices for WNTR.
  """

  def __init__(self, inp_file, important_nodes=[]):
    """Create a WDN using an EPANET .inp file.

    Args:
      inp_file (String): The file path of the .inp file.
      importnat_nodes (iterable of nodes): List of nodes (Strings) interesting for ML and plotting.
    """
    self.network = wntr.network.WaterNetworkModel(inp_file)
    self.important_nodes = self.network.get_graph().nodes()
    if important_nodes:
      self.important_nodes = list(set.intersection(set(self.important_nodes), set(important_nodes)))

  def get_network(self):
    """Return the underlying wntr network.
    """
    return self.network
  
  def show(self):
    """Print nodes and edges of the network and plot it as a graph.
    """
    print(f"Nodes: {sorted(self.network.get_graph().nodes())}")
    print(f"->Important: {sorted(self.important_nodes)}")
    print(f"Edges: {self.network.get_graph().edges()}")
    wntr.graphics.plot_network(self.network, title=self.network.name, node_labels=True, link_labels=True)

  def simulate(self, time_in_hours, leakages=[]):
    """Runs a simulation on the network considering leakages.

    Args:
      time_in_hours (Int): Duration of the simulation in hours.
      leakages (iterable of leakages): Leakages consisting of the node id, the affected area and the start and end time in hours.
    
    Returns:
      Pandas Dataframe with the hour as index and pressure values for every node.
    """

    # Prepare simulation
    self.network.reset_initial_values()
    self.network.options.time.duration = time_in_hours*3600

    # Add leakages
    for n_id in self.important_nodes:
      self.network.get_node(n_id).remove_leak(self.network)
    for n_id, area, time_start_in_hours, time_end_in_hours in leakages:
      self.network.get_node(n_id).add_leak(self.network, area=area, start_time=time_start_in_hours*3600, end_time=time_end_in_hours*3600)

    # Run simulation
    sim = wntr.sim.WNTRSimulator(self.network)
    results = sim.run_sim()

    # Shape results
    pressures = results.node["pressure"]
    pressures['hour'] = list(range(pressures.shape[0]))
    pressures['hour of the day'] = pressures['hour'] % 24
    pressures['day'] = pressures['hour'] // 24
    pressures = pressures.set_index('hour')

    return pressures
