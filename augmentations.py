class RandomRotationAndTranslation(object):
    def __init__(self, max_angle=180, max_translation=1.0):
        self.max_angle = max_angle
        self.max_translation = max_translation

    def __call__(self, sample):
        if 'graph' not in sample:
            print("Warning: 'graph' key not found in the sample.")
            return sample

        protein_data = sample['graph']
        if not isinstance(protein_data, dict) or 'node_position' not in protein_data:
            print(f"Warning: Unexpected format for 'graph'. Expected a dictionary with 'node_position', but got {protein_data}.")
            return sample

        label = sample['label']

        # Checking if 'node_position' is present and is a tensor
        if 'node_position' in protein_data and isinstance(protein_data['node_position'], torch.Tensor):
            # Random rotation
            rotation_matrix = self.random_rotation_matrix(self.max_angle)
            protein_data['node_position'] = self.rotate_coordinates(protein_data['node_position'], rotation_matrix)

            # Random translation
            translation_vector = self.random_translation_vector(self.max_translation)
            protein_data['node_position'] += torch.tensor(translation_vector, dtype=protein_data['node_position'].dtype)
        else:
            print("Warning: 'node_position' is missing or not a tensor. Skipping rotation and translation.")

        return {'graph': protein_data, 'label': label}


    def random_rotation_matrix(self, max_angle):
        angle = np.random.uniform(-max_angle, max_angle, size=3)
        rotation_matrix = self.euler_to_matrix(angle)
        return rotation_matrix

    def rotate_coordinates(self, coordinates, rotation_matrix):
        return np.dot(coordinates, rotation_matrix.T)

    def euler_to_matrix(self, angles):
        # Converting Euler angles to rotation matrix
        x, y, z = np.radians(angles)
        Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return np.dot(Rz, np.dot(Ry, Rx))

    def random_translation_vector(self, max_translation):
        return np.random.uniform(-max_translation, max_translation, size=3)

