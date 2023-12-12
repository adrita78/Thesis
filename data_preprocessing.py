
import os
import shutil
from collections import Counter
import pandas as pd
from Bio.PDB import PDBParser

folder_path = '/content/gdrive/MyDrive/PGM_Project/Combined_data'
all_files = os.listdir(folder_path)

pdb_files = [file for file in all_files if file.endswith('.pdb')]

labeled_data = []

for pdb_file in pdb_files:
    label = int(pdb_file.split('_')[1])
    entry = {'pdb_file': os.path.join(folder_path, pdb_file), 'label': label}
    labeled_data.append(entry)

# Check if labeled_data contains only dictionaries
if all(isinstance(entry, dict) for entry in labeled_data):
    print("All entries in labeled_data are dictionaries.")
else:
    print("Some entries in labeled_data are not dictionaries.")

folder_path = '/content/gdrive/MyDrive/PGM_Project/Combined_data'
all_files = os.listdir(folder_path)


pdb_files = [file for file in all_files if file.endswith('.pdb')]

labeled_data = []

for pdb_file in pdb_files:

    label = int(pdb_file.split('_')[1])


    labeled_data.append({'pdb_file': os.path.join(folder_path, pdb_file), 'label': label})


print(labeled_data[:5])

def load_pdb_structure(file_path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(file_path, file_path)

first_entry = labeled_data[2]
print(first_entry)


structure = load_pdb_structure(first_entry['pdb_file'])

for model in structure:
    for chain in model:
        for residue in chain:
            for atom in residue:
                print(f'Model: {model.id}, Chain: {chain.id}, Residue: {residue.id}, Atom: {atom.id}')

def generate_classmap_file(labeled_data, classmap_file_path):
    with open(classmap_file_path, 'w') as f:
        for sample in labeled_data:
            pdb_file = sample['pdb_file']
            label = sample['label']
            f.write(f"{pdb_file}\t{label}\n")



classmap_file_path ='/content/gdrive/MyDrive/PGM_Project/classmap.txt'
generate_classmap_file(labeled_data, classmap_file_path)

