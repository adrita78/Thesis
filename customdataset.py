class Protein:
    def __init__(self, edge_list, atom_type_numeric, bond_type, num_node, node_position, atom_name, residue_id, residue_type, residue_feature=None):
        self.edge_list = edge_list
        self.atom_type_numeric = atom_type_numeric
        self.bond_type = bond_type
        self.num_node = num_node
        self.node_position = node_position
        self.atom_name = atom_name
        self.residue_id = residue_id
        self.residue_type = residue_type
        self.residue_feature = residue_feature

    def residue_mask(self, index, compact):
        print(f"Debug: Input index: {index}, type: {type(index)}")
        index = index.tolist()

        print(f"Debug: Converted index to list: {index}, type: {type(index)}")
        mask = torch.zeros_like(self.residue_id, dtype=torch.bool)

        if compact:
            # Assuming `index` is a list of residue indices
            mask[index] = True
        else:
            # Assuming `index` is a list of residue IDs
            for i, rid in enumerate(self.residue_id):
                if rid.item() in index:
                    mask[i] = True

        print(f"Debug: Residue Mask: {mask}, type: {type(mask)}")

        return mask

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

    def load_pdb(self, pdb_file):
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)


        node_position = []
        atom_type = []
        atom_name = []
        residue_id = []

        # Mapping dictionaries
        atom_type_mapping = {}
        atom_name_mapping = {}


        atom_type_numeric = []
        atom_name_numeric = []

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
                        if atom.id not in atom_name_mapping:
                            atom_name_mapping[atom.id] = len(atom_name_mapping)
                        atom_name_numeric.append(atom_name_mapping[atom.id])

                        residue_id.append(residue.id)
                        # atom2residue.append(atom.get_parent().id[1])

        # One-hot encoding atom names
        atom_name_one_hot = F.one_hot(torch.tensor(atom_name_numeric), num_classes=len(atom_name_mapping))


        node_position = torch.tensor(node_position)
        atom_type_numeric = torch.tensor(atom_type_numeric)


        residue_id_numeric = [(int(rid[1]), ord(rid[2]) - ord('A')) for rid in residue_id]
        residue_id = torch.tensor(residue_id_numeric)

        residue_type = [atom.get_parent().id[1] for atom in structure.get_atoms()]
        residue_type_mapping = {rtype: i for i, rtype in enumerate(set(residue_type))}
        residue_type_numeric = torch.tensor([residue_type_mapping[rtype] for rtype in residue_type])
        residue_feature = F.one_hot(residue_type_numeric, num_classes=len(residue_type_mapping))

        num_residue = len(residue_type)


        edge_list = torch.zeros((0, 3), dtype=torch.long)
        bond_type = torch.zeros(0, dtype=torch.long)


        protein = Protein(edge_list, atom_type_numeric, bond_type, num_node=node_position.shape[0],
                          node_position=node_position, atom_name=atom_name_one_hot, residue_id=residue_id,
                          residue_type=residue_type_numeric, residue_feature=residue_feature)

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


    def collate_fn(self, batch):
      graphs = [item['graph'] for item in batch]
      labels = [item['label'] for item in batch]
      edge_list = [graph.edge_list for graph in graphs]
      atom_type_numeric = [graph.atom_type_numeric for graph in graphs]
      bond_type = [graph.bond_type for graph in graphs]
      num_node = [graph.num_node for graph in graphs]
      num_node = [torch.tensor(graph.num_node) if isinstance(graph.num_node, int) else graph.num_node for graph in graphs]
      node_position = [graph.node_position for graph in graphs]
      atom_name = [graph.atom_name for graph in graphs]
      residue_id = [graph.residue_id for graph in graphs]
      residue_type = [graph.residue_type for graph in graphs]
      residue_feature = [graph.residue_feature for graph in graphs]


      edge_list = torch.stack(edge_list)
      atom_type_numeric = torch.stack(atom_type_numeric)
      bond_type = torch.stack(bond_type)
      num_node = torch.stack(num_node)
      node_position = torch.stack(node_position)
      atom_name = torch.stack(atom_name)
      residue_id = torch.stack(residue_id)
      residue_type = torch.stack(residue_type)
      residue_feature = torch.stack(residue_feature)

      labels = torch.tensor(labels)

      return {
        'graph': {
            'edge_list': edge_list,
            'atom_type_numeric': atom_type_numeric,
            'bond_type': bond_type,
            'num_node': num_node,
            'node_position': node_position,
            'atom_name': atom_name,
            'residue_id': residue_id,
            'residue_type': residue_type,
            'residue_feature': residue_feature,
        },
        'label': labels
    }
