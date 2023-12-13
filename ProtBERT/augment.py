def combine(inputs, labels, new_inputs, new_labels):
    new_inputs = np.vstack(new_inputs)
    new_labels = np.hstack(new_labels)

    inputs = np.vstack((inputs, new_inputs))
    labels = np.hstack((labels, new_labels))

    return inputs, labels


def random_replace(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(inputs.shape[0]):
        ip = inputs[idx]
        label = labels[idx]

        try:
            unpadded_len = np.where(ip == 0)[0][0]
        except IndexError:
            unpadded_len = len(ip)
        num_to_replace = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_replace, replace=False)
        ip[indices] = np.random.choice(np.arange(5, 25), num_to_replace, replace=True)

        new_inputs.append(ip)
        new_labels.append(label)

    return new_inputs, new_labels


def random_delete(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(inputs.shape[0]):
        ip = inputs[idx]
        label = labels[idx]

        try:
            unpadded_len = np.where(ip == 0)[0][0]
        except IndexError:
            unpadded_len = len(ip)
        ip = list(ip[:unpadded_len])
        num_to_delete = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_delete, replace=False)
        for i in reversed(sorted(indices)):
            ip.pop(i)
        ip.extend([0] * (200 - len(ip)))

        new_inputs.append(np.asarray(ip))
        new_labels.append(label)

    return new_inputs, new_labels


def random_replace_with_A(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(inputs.shape[0]):
        ip = inputs[idx]
        label = labels[idx]

        try:
            unpadded_len = np.where(ip == 0)[0][0]
        except IndexError:
            unpadded_len = len(ip)
        num_to_replace = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_replace, replace=False)
        ip[indices] = m2['A']

        new_inputs.append(ip)
        new_labels.append(label)

    return new_inputs, new_labels


def random_swap(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(inputs.shape[0]):
        ip = inputs[idx]
        label = labels[idx]

        try:
            unpadded_len = np.where(ip == 0)[0][0]
        except IndexError:
            unpadded_len = len(ip)
        ip = list(ip[:unpadded_len])
        num_to_swap = round(unpadded_len * factor)
        indices = np.random.choice(range(1, unpadded_len, 2), num_to_swap, replace=False)
        for i in indices:
            ip[i-1], ip[i] = ip[i], ip[i-1]
        ip.extend([0] * (200 - len(ip)))

        new_inputs.append(np.asarray(ip))
        new_labels.append(label)

    return new_inputs, new_labels


def random_insertion_with_A(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(inputs.shape[0]):
        ip = inputs[idx]
        label = labels[idx]

        try:
            unpadded_len = np.where(ip == 0)[0][0]
        except IndexError:
            unpadded_len = len(ip)
        ip = list(ip[:unpadded_len])
        num_to_insert = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_insert, replace=False)
        for i in indices:
            ip.insert(i, m2['A'])
        if len(ip) < 200:
            ip.extend([0] * (200 - len(ip)))
        elif len(ip) > 200:
            ip = ip[:200]

        new_inputs.append(np.asarray(ip))
        new_labels.append(label)

    return new_inputs, new_labels


def random_masking(sequences, mask_prob=0.15, mask_token_id=0):
    masked_sequences = np.copy(sequences)
    mask = np.random.rand(*sequences.shape) < mask_prob
    masked_sequences[mask] = mask_token_id
    return masked_sequences
