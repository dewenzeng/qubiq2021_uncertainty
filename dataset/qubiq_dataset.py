import torch
import torchvision
import SimpleITK as sitk
import torchvision.transforms as transforms
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import *
from torch.utils.data.dataset import Dataset
from .dataset_utils import get_train_dataset_path, get_vali_dataset_path, matplotlib_imshow, pad_if_not_square
from .augmentation import *
import matplotlib.pyplot as plt

class QUBIQDataset(Dataset):

    def __init__(self, purpose, args):

        self.purpose = purpose
        if purpose == 'train':
            self.data_dir = get_train_dataset_path(args.dataset, args.train_base_dir)
        else:
            self.data_dir = get_vali_dataset_path(args.dataset, args.vali_base_dir)
        self.images = []
        self.labels = []
        folders = glob(self.data_dir+'/case*')
        folders.sort()
        for folder in folders:
            self.images.append(os.path.join(folder,'image.nii.gz'))
            # since we have multiple labels per image, we need to automatically find how many
            # and load them one by one
            label_fns = glob(folder+'/'+args.task+'*')
            label_fns.sort()
            # num_labels = len(label_fns)
            self.labels.append(label_fns)
        self.patch_size = args.patch_size
        # self.classes = args.classes

    def __getitem__(self, index):
        
        img_itk = sitk.ReadImage(self.images[index])
        img_npy = sitk.GetArrayFromImage(img_itk).squeeze()
        # normalize the image
        img_npy = 255 * (img_npy.astype(np.float) - img_npy.min()) / (img_npy.max() - img_npy.min())
        img_npy = img_npy.astype(np.uint8)
        img_npy = pad_if_not_square(img_npy)
        label_npy_list = []
        for label_fn in self.labels[index]:
            label_itk = sitk.ReadImage(label_fn)
            label_npy = sitk.GetArrayFromImage(label_itk).squeeze().astype(np.uint8)
            label_npy = pad_if_not_square(label_npy)
            label_npy_list.append(label_npy)

        train_transform = Compose([
            AdjustSaturation(0.4),
            AdjustContrast(0.4),
            AdjustBrightness(0.4),
            AdjustHue(0.4),
            RandomTranslate(offset=(0.2, 0.2)),
            RandomRotate(degree=30),
            RandomSizedCrop(size=self.patch_size,scale=(0.9, 1.)),
        ])

        test_transform = Compose([
            Scale(size=self.patch_size),
        ])

        if self.purpose == 'train':
            img_npy, label_npy_list = train_transform(img_npy, label_npy_list)
        else:
            img_npy, label_npy_list = test_transform(img_npy, label_npy_list)

        # print(f'img_npy:{img_npy[0].shape}')
        # print(f'label_npy:{label_npy_list[0].shape}')
        # [print(f'label_npy:{label_npy.shape}') for label_npy in label_npy_list]
        return np.expand_dims(img_npy[0],0), label_npy_list
    
    def prepare_for_contrast(self, index):

        img = Image.open(self.image_files[index]).convert('L')
        pseudo_label = self.pseudo_labels[index]
        date = self.dates[index]

        img = np.asarray(img).astype(np.uint8)
        img = pad_if_not_square(img)
        dummy_label = np.zeros_like(img)

        # this one is the standard transform from paper moco-cxr https://github.com/stanfordmlgroup/MoCo-CXR
        # train_transform = Compose([
        #     RandomHorizontallyFlip(),
        #     RandomRotate(degree=10),
        #     RandomSizedCrop(size=self.patch_size,scale=(0.95, 1.)),
        #     ToTensor(),
        #     Normalize(mean=0.5408, std=0.5172),
        # ])

        train_transform = Compose([
            AdjustSaturation(0.4),
            AdjustContrast(0.4),
            AdjustBrightness(0.4),
            AdjustHue(0.4),
            # AdjustGamma(0.4),
            RandomTranslate(offset=(0.1, 0.1)),
            # RandomHorizontallyFlip(),
            RandomRotate(degree=10),
            RandomSizedCrop(size=self.patch_size,scale=(0.95, 1.)),
            ToTensor(),
            # Normalize(mean=0.5408, std=0.5172),
        ])

        img1, _ = train_transform(img,dummy_label)
        img2, _ = train_transform(img,dummy_label)

        return img1, img2, pseudo_label, date

    def prepare_for_supervised(self, index):

        img = Image.open(self.image_files[index]).convert('L')
        label = Image.open(self.label_files[index])

        img = np.asarray(img).astype(np.uint8)
        label = np.asarray(label).astype(np.uint8)

        # # if use 3 classes
        label[label==1] = 1
        label[label==2] = 2
        label[label==3] = 0

        img = pad_if_not_square(img)
        label = pad_if_not_square(label)

        # if we have data augmentation
        train_transform = Compose([
            AdjustSaturation(0.4),
            AdjustContrast(0.4),
            AdjustBrightness(0.4),
            AdjustHue(0.4),
            RandomTranslate(offset=(0.1, 0.1)),
            RandomRotate(degree=30),
            RandomSizedCrop(size=self.patch_size,scale=(0.95, 1.)),
            ToTensor(),
        ])

        # if we do not use any data augmentation
        # train_transform = Compose([
        #     Scale(size=self.patch_size),
        #     ToTensor(),
        # ])

        test_transform = Compose([
            Scale(size=self.patch_size),
            ToTensor()
        ])

        # print(f'img:{img.shape}, label:{label.shape}')
        if self.purpose == 'train':
            img, label = train_transform(img, label)
        else:
            img, label = test_transform(img, label)
        
        # label = (label / 255.) * (self.classes - 1)

        return img, label

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_base_dir", type=str, default="d:/data/QUBIQ2021/training_data_v3/training_data_v3")
    parser.add_argument("--vali_base_dir", type=str, default="d:/data/QUBIQ2021/validation_data_qubiq2021/validation_data_qubiq2021")
    parser.add_argument("--dataset", type=str, default='prostate')
    parser.add_argument("--task", type=str, default='task01')
    parser.add_argument("--patch_size", type=int, default='512')
    args = parser.parse_args()
    train_dataset = QUBIQDataset('train',args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=5,shuffle=True,num_workers=1,drop_last=False)
    for batch_idx, tup in enumerate(train_dataloader):
        img, label_list = tup
        print(f'img:{img.shape}')
        plt.figure(1)
        img_grid = torchvision.utils.make_grid(img)
        matplotlib_imshow(img_grid, one_channel=False)
        plt.figure(2)
        img_grid = torchvision.utils.make_grid(label_list[0].unsqueeze(dim=1))
        matplotlib_imshow(img_grid, one_channel=False)
        plt.figure(3)
        img_grid = torchvision.utils.make_grid(label_list[1].unsqueeze(dim=1))
        matplotlib_imshow(img_grid, one_channel=False)
        # plt.show()
        # break