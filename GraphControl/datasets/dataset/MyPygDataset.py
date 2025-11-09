import torch
from torch_geometric.data import InMemoryDataset

class MyPygDataset(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])
        self.slices = None
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['H1N1_graph_2012.pt']

    def download(self):
        pass

    def process(self):
        pass