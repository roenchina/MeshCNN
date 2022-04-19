import os
import torch
import numpy as np
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh


# TODO ModalData
class ModalData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot


        # data_root = /sherc_modal
        # self.dir = /sherc_modal/test
        # eigen_dir = /sherc_modal/eigen
        self.dir = os.path.join(opt.dataroot, opt.phase)

        # paths = [mesh_full_path] e.g. ['/sherc_modal/test/T0.obj']
        self.paths = self.make_dataset(self.dir)

        # eigen_paths = [eigen_path_of_each_mesh] e.g. ['/sherc_modal/eigen/T0']
        self.eigen_paths = self.get_eigen_paths(self.paths, os.path.join(opt.dataroot, 'eigen'))

        # TODO what is classes? is it nessessary?
        self.classes = [1]
        self.nclasses = len(self.classes)

        self.size = len(self.paths)

        self.get_mean_std()

        # TODO what is this for?
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels


    def __getitem__(self, index):
        path = self.paths[index]
        mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)

        meta = {}
        meta['mesh'] = mesh

        # TODO why do we need to pad?
        label = read_label(self.eigen_paths[index])
        meta['label'] = label

        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std

        return meta

    def __len__(self):
         return self.size

    @staticmethod
    def get_eigen_paths(paths, eigen_dir):
        eigen_paths = []
        for path in paths:
            data_name = os.path.splitext(os.path.basename(path))[0]
            eigen = os.path.join(eigen_dir, data_name)
            assert(os.path.isdir(eigen))
            eigen_paths.append(eigen)
        return eigen_paths

    @staticmethod
    def make_dataset(path):
        meshes = []
        assert os.path.isdir(path), '%s is not a valid directory' % path
        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if is_mesh_file(fname):
                    path = os.path.join(root, fname)
                    meshes.append(path)
        return meshes


def read_label(label_dir):
    # label_dir = 'shrec/eigen/T0'
    # label_file = 'shrec/eigen/T0/label.npy'
    label_file = os.path.join(label_dir, 'label.npy')
    label = np.load(label_file)
    return label