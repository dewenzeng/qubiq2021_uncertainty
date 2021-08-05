import warnings
warnings.filterwarnings("ignore")
import os
from datetime import datetime
import time
import torch
import random
from network.unet2d import UNet2D
from dataset.qubiq_dataset import QUBIQDataset
import torch.nn.functional as F
from myconfig import get_config
from batchgenerators.utilities.file_and_folder_operations import *
from lr_scheduler import LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
from experiment_log import PytorchExperimentLogger
from metrics import SegmentationMetric

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def run():
    # initialize config
    args = get_config()
    args.experiment_name = args.dataset+'_'+args.task+'_'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.save_path = os.path.join(args.results_dir, args.experiment_name + args.save)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    writer = SummaryWriter('runs/' + args.experiment_name + args.save)
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
    model_result_dir = join(args.save_path, 'model')
    maybe_mkdir_p(model_result_dir)
    if not os.path.exists(model_result_dir):
        os.mkdir(model_result_dir)
    args.model_result_dir = model_result_dir
    # create model
    logger.print("creating model ...")
    model = UNet2D(in_channels=1, initial_filter_size=args.initial_filter_size, kernel_size=3, classes=args.classes, do_instancenorm=True, dropout=0)
    if args.restart:
        logger.print('loading from saved model ' + args.pretrained_model_path)
        dict = torch.load(args.pretrained_model_path,
                            map_location=lambda storage, loc: storage)
        save_model = dict["net"]
        model.load_state_dict(save_model)
    model.to(args.device)

    train_dataset = QUBIQDataset(purpose='train', args=args)
    logger.print('training data dir '+train_dataset.data_dir)
    validate_dataset = QUBIQDataset(purpose='val', args=args)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader), min_lr=args.min_lr)
    best_dice = 0
    for epoch in range(args.epochs):
        # train for one epoch
        train_loss = train(train_loader, model, criterion, epoch, optimizer, scheduler, logger, args)
        writer.add_scalar('training_loss', train_loss, epoch)
        writer.add_scalar('learning_rate_fold', optimizer.param_groups[0]['lr'], epoch)
        if (epoch % 2 == 0):
            # evaluate for one epoch
            val_loss, val_dice = validate(validate_loader, model, criterion, epoch, logger, args)

            logger.print('Validation Epoch: {0}\t'
                            'Training Loss {val_loss:.4f} \t'
                            'Validation Dice {val_dice:.4f} \t'
                            .format(epoch + 1, val_loss=val_loss, val_dice=val_dice))

            # results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss)
            # results.save()
            if best_dice < val_dice:
                best_dice = val_dice
            writer.add_scalar('validate_dice', val_dice, epoch)
            writer.add_scalar('best_dice', best_dice, epoch)
            # save model
            save_dict = {"net": model.state_dict()}
            torch.save(save_dict, os.path.join(args.model_result_dir, "latest.pth"))

def train(data_loader, model, criterion, epoch, optimizer, scheduler, logger, args):
    model.train()
    # metric_val = SegmentationMetric(args.classes)
    # metric_val.reset()
    losses = AverageMeter()
    for batch_idx, tup in enumerate(data_loader):
        img, label_list = tup
        image_var = img.float().to(args.device)
        label_list = [label.long().to(args.device) for label in label_list]
        scheduler(optimizer, batch_idx, epoch)
        x_out = model(image_var)
        # Do softmax
        # x_out = F.softmax(x_out, dim=1)
        # get the average cross entropy as the loss function
        loss_list = [criterion(x_out, label) for label in label_list]
        loss = torch.stack(loss_list).mean(dim=0)
        # print(f'out.min:{x_out.min()}, out.max:{x_out.max()}')
        # loss = criterion(x_out, label.squeeze())
        losses.update(loss.item(), image_var.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # no evaluation on the training data
        # metric_val.update(label, x_out)
        # _, _, Dice = metric_val.get()
        logger.print(f"Training epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, lr:{optimizer.param_groups[0]['lr']:.7f}, loss:{losses.avg:.4f}")
    # pixAcc, mIoU, mDice = metric_val.get()
    return losses.avg

def validate(data_loader, model, criterion, epoch, logger, args):
    model.eval()
    metric_val = SegmentationMetric(args.classes)
    metric_val.reset()
    losses = AverageMeter()
    threshold_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    with torch.no_grad():
        for batch_idx, tup in enumerate(data_loader):
            img, label_list = tup
            image_var = img.float().to(args.device)
            label_list = [label.long().to(args.device) for label in label_list]
            x_out = model(image_var)
            loss_list = [criterion(x_out, label) for label in label_list]
            loss = torch.stack(loss_list).mean(dim=0)
            losses.update(loss.item(), image_var.size(0))
            x_out = F.softmax(x_out, dim=1)[:,-1,:,:]
            for threhold in threshold_list:
                for label in label_list:
                    metric_val.update(label.cpu().numpy(), x_out.cpu().numpy()>threhold)
            pixAcc, mIoU, Dice = metric_val.get()
            logger.print(f"Validation epoch:{epoch}, batch:{batch_idx}/{len(data_loader)}, mean Dice:{Dice:3f}")
            # save result
            # if batch_idx == 10 and epoch % 20 == 0:
            #     # normalize image
            #     image_var = (image_var - image_var.min()) / (image_var.max() - image_var.min())
            #     img_grid = torchvision.utils.make_grid(image_var.data.cpu())
            #     writer.add_image('val_epoch_fold'+str(fold) + '_' + str(epoch) + '_imgs', img_grid)
            #     predictions = torch.argmax(x_out.data.cpu(), dim=1).unsqueeze(dim=1)
            #     predictions = (predictions.float() - predictions.min()) / (predictions.max() - predictions.min())
            #     img_grid = torchvision.utils.make_grid(predictions)
            #     writer.add_image('val_epoch_fold'+str(fold) + '_' + str(epoch) + '_predictions', img_grid)
            #     label = (label.float() - label.min()) / (label.max() - label.min())
            #     img_grid = torchvision.utils.make_grid(label)
            #     writer.add_image('val_epoch_fold'+str(fold) + '_' + str(epoch) + '_label', img_grid)
    pixAcc, mIoU, mDice = metric_val.get()
    return losses.avg, mDice

if __name__ == '__main__':
    run()