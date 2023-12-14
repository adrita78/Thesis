import os
import torch
import warnings
from tqdm import tqdm
from torch.utils import data as torch_data
from torchdrug import data, utils
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from torchdrug.core import Registry as R
from Bio.PDB import PDBParser
from torch.utils.data.dataloader import default_collate
import numpy as np
import torch.nn.functional as F
from Bio.PDB import NeighborSearch
from Bio.PDB import Atom
from torch.nn.utils.rnn import pad_sequence

class CustomProteinDataset(Dataset):
    def __init__(self, labeled_data, transform=None, lazy=False, verbose=0):
        self.data = labeled_data
        #self.classmap = classmap
        self.transform = transform
        self.lazy = lazy
        self.verbose = verbose
        self.loaded_data = []
        self.pdb_files = []
        self.targets = []

        label_list = self.get_label_list(labeled_data)


        if self.verbose:
            labeled_data_iter = tqdm(labeled_data, desc="Constructing proteins from pdbs")
        else:
            labeled_data_iter = labeled_data

        for i, sample in enumerate(labeled_data_iter):
            pdb_file = sample['pdb_file']
            label = label_list.get(pdb_file, -1)
            protein = self.load_pdb(pdb_file)

            # if self.augmentations is not None:
            #     for augmentation in self.augmentations:
            #         protein = augmentation(protein)

            if self.lazy and i != 0:
                protein = None
            if hasattr(protein, "residue_feature") and protein.residue_feature is not None:
                protein.residue_feature = protein.residue_feature.to_sparse()

            self.loaded_data.append(protein)
            self.pdb_files.append(pdb_file)
            self.targets.append(label)

    def __len__(self):
        return len(self.loaded_data)
    def load_pdb(self, pdb_file, max_num_atoms=None, atom_name_mapping=None):
      parser = PDB.PDBParser(QUIET=True)
      structure = parser.get_structure("protein", pdb_file)

      node_position = []
      atom_type_numeric = []
      edge_list = []
      bond_type = []

    # Mapping dictionaries
      atom_type_mapping = {}
      if atom_name_mapping is None:
        atom_name_mapping = {}

      residue_id = []

    # Extracting information from the structure
      for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    node_position.append(atom.coord)

                    # Atom type mapping
                    if atom.element not in atom_type_mapping:
                        atom_type_mapping[atom.element] = len(atom_type_mapping)
                    atom_type_numeric.append(atom_type_mapping[atom.element])

                    # Atom name mapping
                    atom_name = atom.id
                    if atom_name not in atom_name_mapping:
                        atom_name_mapping[atom_name] = len(atom_name_mapping)

                    residue_id.append(residue.id)

      # Extracting bond information using Biopython
      for model in structure:
        for chain in model:
          for residue in chain:
            for atom in residue:
                neighbors = list(atom.get_parent().get_atoms())
                neighbors.remove(atom)

                for bonded_atom in neighbors:
                    if type(bonded_atom) is Atom:
                        #print(f"Bonded Atom ID: {bonded_atom.id}")
                        #print(f"Bond Order: {bonded_atom.order}")
                        edge_list.append([atom.id, bonded_atom.id])
                        bond_type.append(bonded_atom.order) if hasattr(bonded_atom, 'order') else 1

      print(f'Atom ID: {atom.id}')
      print(f"Bonded Atom ID: {bonded_atom.id}")
      print(f"Bond Order (using 'element'): {bonded_atom.element}")
      #print("Edge List:", edge_list)
      #print("Bond Type:", bond_type)

    # One-hot encoding atom names
      atom_name_numeric = [atom.id for atom in structure.get_atoms()]
      atom_name_indices = [atom_name_mapping[atom.id] for atom in structure.get_atoms()]
      atom_name_one_hot = F.one_hot(torch.tensor(atom_name_indices), num_classes=len(atom_name_mapping))

      node_position = torch.tensor(node_position)
      atom_type_numeric = torch.tensor(atom_type_numeric)
      edge_list = torch.tensor(edge_list, dtype=torch.long)
      bond_type = torch.tensor(bond_type, dtype=torch.long)

    # Padding
      if max_num_atoms is not None:
        num_atoms = node_position.size(0)
        if num_atoms < max_num_atoms:
            padding_size = max_num_atoms - num_atoms
            zero_padding = torch.zeros(padding_size, 3)
            node_position = torch.cat([node_position, zero_padding])

            atom_type_padding = torch.zeros(padding_size)
            atom_type_numeric = torch.cat([atom_type_numeric, atom_type_padding])

            atom_name_padding = torch.zeros(padding_size, len(atom_name_mapping))
            atom_name_one_hot = torch.cat([atom_name_one_hot, atom_name_padding])

      residue_id_numeric = [(int(rid[1]), ord(rid[2]) - ord('A')) for rid in residue_id]
      residue_id = torch.tensor(residue_id_numeric)

      residue_type = [atom.get_parent().id[1] for atom in structure.get_atoms()]
      residue_type_mapping = {rtype: i for i, rtype in enumerate(set(residue_type))}
      residue_type_numeric = torch.tensor([residue_type_mapping[rtype] for rtype in residue_type])
      residue_feature = F.one_hot(residue_type_numeric, num_classes=len(residue_type_mapping))

      num_residue = len(residue_type)

      protein = data.Protein(
        edge_list=edge_list,
        atom_type=atom_type_numeric,
        bond_type=bond_type,
        num_node=node_position.shape[0],
        node_position=node_position,
        atom_name=atom_name_one_hot,
        residue_id=residue_id,
        residue_type=residue_type_numeric,
        residue_feature=residue_feature
    )

      protein.residue = lambda: None
      return protein


    def __getitem__(self, index):
        protein_data = self.loaded_data[index]
        label = self.targets[index]


        item = {'graph': protein_data, 'label': label}
        if self.transform:
            item = self.transform(item)

        return item


    def get_label_list(self, labeled_data):
        label_list = {}

        for sample in labeled_data:
            pdb_file = sample['pdb_file']
            label = sample.get('label', -1)
            label_list[pdb_file] = label

        return label_list

    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
      graphs = [item['graph'] for item in batch]
      labels = [item['label'] for item in batch]

    # Extracting individual components from each graph
      edge_lists = [graph.edge_list for graph in graphs]
      atom_types = [graph.atom_type for graph in graphs]
      bond_types = [graph.bond_type for graph in graphs]
      #num_nodes = [graph.num_node for graph in graphs]
      num_nodes = [graph.num_node.item() for graph in graphs]

      #num_nodes = sum(graph.num_node for graph in graphs)
      node_positions = [graph.node_position for graph in graphs]
      atom_names = [graph.atom_name for graph in graphs]
      residue_ids = [graph.residue_id for graph in graphs]
      residue_types = [graph.residue_type for graph in graphs]
      residue_features = [graph.residue_feature for graph in graphs]

    # Padding sequences to a common length
      edge_lists = [torch.tensor(edge_list, dtype=torch.long) for edge_list in edge_lists]
      edge_lists = pad_sequence(edge_lists, batch_first=True)
      num_nodes = torch.tensor(num_nodes)


      atom_types = pad_sequence(atom_types, batch_first=True)
      bond_types = pad_sequence(bond_types, batch_first=True)
      node_positions = pad_sequence(node_positions, batch_first=True)
    
    # Padding the inner dimension of atom_names individually
      max_atom_name_length = max(atom_name.size(0) for atom_name in atom_names)
      max_atom_name_dim2 = max(atom_name.size(1) for atom_name in atom_names)

      atom_names_padded = torch.zeros(len(atom_names), max_atom_name_length, max_atom_name_dim2)
      for i, atom_name in enumerate(atom_names):
        atom_name_length = min(atom_name.size(0), max_atom_name_length)
        atom_names_padded[i, :atom_name_length, :atom_name.size(1)] = atom_name[:atom_name_length, :]
  
      residue_ids = pad_sequence(residue_ids, batch_first=True)
      residue_types = pad_sequence(residue_types, batch_first=True)
      # Finding the maximum size of residue_features in the batch
      max_residue_features_size = max(graph.residue_feature.size(0) for graph in graphs)
      residue_features = [graph.residue_feature.to_dense() for graph in graphs]
      max_residue_feature_size = max(residue_feature.size(1) for residue_feature in residue_features)
      residue_features_padded = [
    F.pad(residue_feature, (0, max_residue_feature_size - residue_feature.size(1)))
    for residue_feature in residue_features
]
      residue_features_padded = pad_sequence(residue_features_padded, batch_first=True)


      labels = torch.tensor(labels)

      return {
        'graph': {
            'edge_list': edge_lists,
            'atom_type': atom_types,
            'bond_type': bond_types,
            'num_node': num_nodes,
            'node_position': node_positions,
            'atom_name': atom_names_padded,
            'residue_id': residue_ids,
            'residue_type': residue_types,
            'residue_feature': residue_features,
        },
        'label': labels
    }
