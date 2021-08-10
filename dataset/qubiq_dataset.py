from numpy import float64
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

# can only handle training and validtion set cause this class read the image and label at the same time
class QUBIQDataset(Dataset):

    def __init__(self, purpose, args):

        self.purpose = purpose
        self.dataset = args.dataset
        if purpose == 'train':
            self.data_dir = get_train_dataset_path(args.dataset, args.train_base_dir)
        else:
            self.data_dir = get_vali_dataset_path(args.dataset, args.vali_base_dir)
        self.images = []
        self.labels = []
        folders = glob(self.data_dir+'/case*')
        folders.sort()
        # if we are dealing with 3D images, we split them into 2D slices
        if args.dataset in ['pancreas', 'pancreatic-lesion']:
            for folder in folders:
                # read image
                img_itk = sitk.ReadImage(os.path.join(folder,'image.nii.gz'))
                img_npy = sitk.GetArrayFromImage(img_itk).squeeze()
                img_npy = img_npy.transpose(2,0,1)
                img_npy[img_npy<-1500] = -1500
                img_npy = 255 * (img_npy.astype(np.float64) - img_npy.min()) / (img_npy.max() - img_npy.min())
                img_npy = img_npy.astype(np.uint8)
                # read labels
                label_fns = glob(folder+'/'+args.task+'*')
                label_fns.sort()
                tmp_label_list = []
                # let's get the min and max slice where there are labels
                label_sums = []
                for label_fn in label_fns:
                    label_itk = sitk.ReadImage(label_fn)
                    label_npy = sitk.GetArrayFromImage(label_itk).squeeze().astype(np.uint8)
                    label_npy = label_npy.transpose(2,0,1)
                    tmp_label_list.append(label_npy)
                    label_sums.append(label_npy.sum(axis=(1,2)))
                label_sums = np.stack(label_sums,axis=0).mean(axis=0)
                min_slice = max(np.where(label_sums == label_sums[label_sums>0][0])[0][0]-5, 0)
                max_slice = min(np.where(label_sums == label_sums[label_sums>0][-1])[0][0]+5, tmp_label_list[0].shape[0]-1)
                # print(f'min_slice:{min_slice}, max_slice:{max_slice}')
                # for slice_idx in range(img_npy.shape[0]):
                for slice_idx in range(min_slice, max_slice):
                    self.images.append([pad_if_not_square(img_npy[slice_idx])])
                    self.labels.append([pad_if_not_square(label[slice_idx]) for label in tmp_label_list])
        else:
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

        # this is for prostate dataset
        # train_transform = Compose([
        #         AdjustSaturation(0.2),
        #         AdjustContrast(0.2),
        #         AdjustBrightness(0.2),
        #         AdjustHue(0.2),
        #         RandomTranslate(offset=(0.2, 0.2)),
        #         RandomRotate(degree=10),
        #         RandomSizedCrop(size=self.patch_size,scale=(0.9, 1.)),
        #     ])
        
        train_transform = Compose([
                AdjustSaturation(0.2),
                AdjustContrast(0.2),
                AdjustBrightness(0.2),
                AdjustHue(0.2),
                RandomTranslate(offset=(0.1, 0.1)),
                # RandomRotate(degree=10),
                RandomSizedCrop(size=self.patch_size,scale=(0.9, 1.)),
            ])

        test_transform = Compose([
            Scale(size=self.patch_size),
        ])

        if self.dataset in ['pancreas', 'pancreatic-lesion']:
            img_npy = self.images[index]
            label_npy_list = self.labels[index]
        else:
            img_itk = sitk.ReadImage(self.images[index])
            img_npy = sitk.GetArrayFromImage(img_itk).squeeze()
            if self.dataset == 'kidney':
                img_npy[img_npy<-400] = -400
                img_npy[img_npy>2000] = 2000
            # normalize the image
            img_npy = 255 * (img_npy.astype(np.float) - img_npy.min()) / (img_npy.max() - img_npy.min())
            img_npy = img_npy.astype(np.uint8)
            if len(img_npy.shape) == 3:
                img_npy = [pad_if_not_square(img_npy[i]) for i in range(img_npy.shape[0])]
            else:
                img_npy = [pad_if_not_square(img_npy)]
            label_npy_list = []
            for label_fn in self.labels[index]:
                label_itk = sitk.ReadImage(label_fn)
                label_npy = sitk.GetArrayFromImage(label_itk).squeeze().astype(np.uint8)
                label_npy = pad_if_not_square(label_npy)
                label_npy_list.append(label_npy)
            # print(f'img_npy:{img_npy.shape}')
            # print(f'label_npy:{label_npy_list[0].shape}')
            # [print(f'label_npy:{label_npy.shape}') for label_npy in label_npy_list]
        if self.purpose == 'train':
            img_npy, label_npy_list = train_transform(img_npy, label_npy_list)
        else:
            img_npy, label_npy_list = test_transform(img_npy, label_npy_list)

        img_npy = np.stack(img_npy,0)
        # img_npy = np.expand_dims(img_npy,0)
        img_npy = img_npy.astype(float64) / 255
        return img_npy, label_npy_list
    
    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_base_dir", type=str, default="d:/data/QUBIQ2021/training_data_v3/training_data_v3")
    parser.add_argument("--vali_base_dir", type=str, default="d:/data/QUBIQ2021/validation_data_qubiq2021/validation_data_qubiq2021")
    parser.add_argument("--dataset", type=str, default='pancreatic-lesion')
    parser.add_argument("--task", type=str, default='task01')
    parser.add_argument("--patch_size", type=int, default='512')
    args = parser.parse_args()
    train_dataset = QUBIQDataset('train',args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=5,shuffle=True,num_workers=1,drop_last=False)
    for batch_idx, tup in enumerate(train_dataloader):
        img, label_list = tup
        print(f'img:{img.shape}')
        print(f'img max:{img.max()}')
        print(f'label max:{label_list[0].max()}')
        plt.figure(1)
        img_grid = torchvision.utils.make_grid(img)
        matplotlib_imshow(img_grid, one_channel=False)
        plt.figure(2)
        img_grid = torchvision.utils.make_grid(label_list[0].unsqueeze(dim=1))
        matplotlib_imshow(img_grid, one_channel=False)
        plt.figure(3)
        img_grid = torchvision.utils.make_grid(label_list[1].unsqueeze(dim=1))
        matplotlib_imshow(img_grid, one_channel=False)
        plt.show()
        break