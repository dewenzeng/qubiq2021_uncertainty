import warnings
warnings.filterwarnings("ignore")
import os
from datetime import datetime
import torch
import SimpleITK as sitk
from glob import glob
from network.unet2d import UNet2D
import torch.nn.functional as F
from myconfig import get_config
from batchgenerators.utilities.file_and_folder_operations import *
from lr_scheduler import LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
from experiment_log import PytorchExperimentLogger
from metrics import SegmentationMetric
from dataset.dataset_utils import pad_if_not_square, get_vali_dataset_path
from dataset.augmentation import *
from skimage.transform import resize

def run():
    # initialize config
    args = get_config()
    args.experiment_name = args.dataset+'_'+args.task+'_'+'predict_'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.save_path = os.path.join(args.results_dir, args.experiment_name + args.save)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger = PytorchExperimentLogger(args.save_path, "elog", ShowTerminal=True)
    # setup cuda
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.print(f"the model will run on device:{args.device}")
    torch.manual_seed(args.seed)
    if 'cuda' in str(args.device):
        torch.cuda.manual_seed_all(args.seed)
    image_result_dir = join(args.save_path, 'images')
    if not os.path.exists(image_result_dir):
        os.mkdir(image_result_dir)
    args.image_result_dir = image_result_dir
    # create model
    logger.print("creating model ...")
    model = UNet2D(in_channels=1, initial_filter_size=args.initial_filter_size, kernel_size=3, classes=args.classes, do_instancenorm=True, dropout=0)
    logger.print('loading from saved model ' + args.pretrained_model_path)
    dict = torch.load(args.pretrained_model_path,
                        map_location=lambda storage, loc: storage)
    save_model = dict["net"]
    model.load_state_dict(save_model)
    model.to(args.device)

    # load the test data on the fly
    images = []
    folder_names = []
    data_dir = get_vali_dataset_path(args.dataset, args.test_base_dir)
    logger.print(f'data_dir:{data_dir}')
    folders = glob(data_dir+'/case*')
    folders.sort()
    for folder in folders:
        images.append(os.path.join(folder,'image.nii.gz'))
        without_extra_slash = os.path.normpath(folder)
        os.mkdir(os.path.join(args.image_result_dir, os.path.basename(without_extra_slash)))
        folder_names.append(os.path.basename(without_extra_slash))
    
    test_transform = Compose([
        Scale(size=args.patch_size),
    ])

    print(f'folder_names:{folder_names}')
    model.eval()
    with torch.no_grad():
        for index in range(len(images)):
            if args.dataset in ['pancreas', 'pancreatic-lesion']:
                img_itk = sitk.ReadImage(images[index])
                img_npy = sitk.GetArrayFromImage(img_itk).squeeze()
                img_npy = img_npy.transpose(2,0,1)
                img_npy[img_npy<-1500] = -1500
                img_npy = 255 * (img_npy.astype(np.float64) - img_npy.min()) / (img_npy.max() - img_npy.min())
                img_npy = img_npy.astype(np.uint8)
                out_list = []
                for slice_idx in range(img_npy.shape[0]):
                    img_npy_slice = img_npy[slice_idx]
                    original_shape = img_npy_slice.shape
                    img_npy_slice = pad_if_not_square(img_npy_slice)
                    original_shape_squared = img_npy_slice.shape
                    fake_label_npy = np.ones(img_npy_slice.shape)
                    img_npy_slice, label_npy_list = test_transform(img_npy_slice, [fake_label_npy])
                    img_npy_slice = np.expand_dims(img_npy_slice[0],0)
                    img_torch = torch.from_numpy(img_npy_slice)[None].float().to(args.device)
                    x_out = model(img_torch)
                    x_out = F.softmax(x_out, dim=1)[0,-1,:,:].cpu().numpy()
                    out_resized = resize(x_out, (original_shape_squared[0], original_shape_squared[1]), anti_aliasing=True)
                    out_resized = out_resized[int(out_resized.shape[0]/2.)-int(original_shape[0]/2.):int(out_resized.shape[0]/2.)+int(original_shape[0]/2.), int(out_resized.shape[1]/2.)-int(original_shape[1]/2.):int(out_resized.shape[1]/2.)+int(original_shape[1]/2.)]
                    out_list.append(out_resized)
                out_resized = np.stack(out_list,0)
                out_resized = out_resized.transpose(1,2,0)
                out_resized_itk = sitk.GetImageFromArray(out_resized)
                out_resized_itk.SetOrigin(img_itk.GetOrigin())
                out_resized_itk.SetDirection(img_itk.GetDirection())
                out_resized_itk.SetSpacing(img_itk.GetSpacing())
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(args.image_result_dir, folder_names[index], args.task+'.nii.gz'))
                writer.Execute(out_resized_itk)
            else:
                img_itk = sitk.ReadImage(images[index])
                # print(f'img_npy:{sitk.GetArrayFromImage(img_itk).shape}')
                img_npy = sitk.GetArrayFromImage(img_itk).squeeze()
                if len(img_npy.shape) == 3:
                    img_npy = img_npy[0]
                original_shape = img_npy.shape
                # normalize the image
                img_npy = 255 * (img_npy.astype(np.float) - img_npy.min()) / (img_npy.max() - img_npy.min())
                img_npy = img_npy.astype(np.uint8)
                img_npy = pad_if_not_square(img_npy)
                original_shape_squared = img_npy.shape
                fake_label_npy = np.ones(img_npy.shape)
                img_npy, label_npy_list = test_transform(img_npy, [fake_label_npy])
                img_npy = np.expand_dims(img_npy[0],0)
                img_torch = torch.from_numpy(img_npy)[None].float().to(args.device)
                x_out = model(img_torch)
                x_out = F.softmax(x_out, dim=1)[0,-1,:,:].cpu().numpy()
                # scale the prediction to the original size squared
                out_resized = resize(x_out, (original_shape_squared[0], original_shape_squared[1]), anti_aliasing=True)
                # crop to the orginal size
                out_resized = out_resized[int(out_resized.shape[0]/2.)-int(original_shape[0]/2.):int(out_resized.shape[0]/2.)+int(original_shape[0]/2.), int(out_resized.shape[1]/2.)-int(original_shape[1]/2.):int(out_resized.shape[1]/2.)+int(original_shape[1]/2.)]
                # save this image
                out_resized_itk = sitk.GetImageFromArray(out_resized)
                # out_resized_itk = sitk.GetImageFromArray(np.expand_dims(out_resized, 0))
                out_resized_itk.SetOrigin(img_itk.GetOrigin())
                out_resized_itk.SetDirection(img_itk.GetDirection())
                out_resized_itk.SetSpacing(img_itk.GetSpacing())
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(args.image_result_dir, folder_names[index], args.task+'.nii.gz'))
                writer.Execute(out_resized_itk)

if __name__ == '__main__':
    run()