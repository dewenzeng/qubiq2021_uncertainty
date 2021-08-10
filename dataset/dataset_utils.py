import os
import numpy as np
import matplotlib.pyplot as plt

def pad_if_too_small(data, sz):
    reshape = (len(data.shape) == 2)
    if reshape:
        h, w = data.shape
        data = data.reshape((h, w, 1))

    h, w, c = data.shape

    if not (h >= sz and w >= sz):
        # img is smaller than sz
        # we are missing by at least 1 pixel in at least 1 edge
        new_h, new_w = max(h, sz), max(w, sz)
        new_data = np.zeros([new_h, new_w, c], dtype=data.dtype)

        # will get correct centre, 5 -> 2
        centre_h, centre_w = int(new_h / 2.), int(new_w / 2.)
        h_start, w_start = centre_h - int(h / 2.), centre_w - int(w / 2.)

        new_data[h_start:(h_start + h), w_start:(w_start + w), :] = data
    else:
        new_data = data
        new_h, new_w = h, w

    if reshape:
        new_data = new_data.reshape((new_h, new_w))

    return new_data

def pad_if_not_square(orig_data):

    w, h = orig_data.shape
    if w == h:
        return orig_data
    elif w > h:
        return pad_if_too_small(orig_data,w)
    else:
        return pad_if_too_small(orig_data,h)

def get_train_dataset_path(dataset, base_dir):
    if dataset == 'brain-growth':
        dataset_path = os.path.join(base_dir, 'brain-growth', 'Training')
    elif dataset == 'brain-tumor':
        dataset_path = os.path.join(base_dir, 'brain-tumor', 'Training')
    elif dataset == 'kidney':
        dataset_path = os.path.join(base_dir, 'kidney', 'Training')
    elif dataset == 'pancreas':
        dataset_path = os.path.join(base_dir, 'pancreas')
    elif dataset == 'pancreatic-lesion':
        dataset_path = os.path.join(base_dir, 'pancreatic-lesion')
    elif dataset == 'prostate':
        dataset_path = os.path.join(base_dir, 'prostate', 'Training')
    return dataset_path

def get_vali_dataset_path(dataset, base_dir):
    if dataset == 'brain-growth':
        dataset_path = os.path.join(base_dir, 'brain-growth', 'Validation')
    elif dataset == 'brain-tumor':
        dataset_path = os.path.join(base_dir, 'brain-tumor', 'Validation')
    elif dataset == 'kidney':
        dataset_path = os.path.join(base_dir, 'kidney', 'Validation')
    elif dataset == 'pancreas':
        dataset_path = os.path.join(base_dir, 'pancreas', 'Validation')
    elif dataset == 'pancreatic-lesion':
        dataset_path = os.path.join(base_dir, 'pancreatic-lesion', 'Validation')
    elif dataset == 'prostate':
        dataset_path = os.path.join(base_dir, 'prostate', 'Validation')
    return dataset_path

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg)
    else:
        # use this function if image is grayscale
        plt.imshow(npimg[0,:,:],'gray')
        # use this function if image is RGB
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))