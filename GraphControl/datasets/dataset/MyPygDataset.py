import torch
from torch_geometric.data import InMemoryDataset

class MyPygDataset(InMemoryDataset):
    
    def __init__(self, root, year, transform=None, pre_transform=None, pre_filter=None):
        self.year = year
        super().__init__(root, transform, pre_transform, pre_filter)
        print(f'path: {self.processed_paths[0]}')
        self.data = torch.load(self.processed_paths[0], weights_only=False)
        self.slices = None
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'H1N1_graph_{self.year}.pt']

    def download(self):
        pass

    def process(self):
        pass