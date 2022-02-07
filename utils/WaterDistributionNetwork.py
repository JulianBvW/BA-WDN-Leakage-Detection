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
      importnat_nodes (iterable of nodes): List of nodes interesting for ML
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
