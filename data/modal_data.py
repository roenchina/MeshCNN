import os
import torch
import numpy as np
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh


# TODO
class ModalData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

