from __future__ import division
import warnings
from model_mlp_counting import CSRNet
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils_new import save_checkpoint, setup_seed
import torch
import os
import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import numpy as np
from image import load_data
#from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')
import time
import random

#setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')
with_extra_sv = False

def main(args):
    if args['dataset'] == 'ShanghaiA256':
        train_file = './npydata/ShanghaiA256_train.npy'
        test_file = './npydata/ShanghaiA256_test.npy'
        extra_file ='./npydata/ShanghaiB256_train.npy'

    elif args['dataset'] == 'ShanghaiB256':
        train_file = './npydata/ShanghaiB256_train.npy'
        test_file = './npydata/ShanghaiB256_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/UCF_QNRF256_train.npy'
        test_file = './npydata/UCF_QNRF256_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/JHU256_train.npy'
        test_file = './npydata/JHU256_test.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    if with_extra_sv:
        with open(extra_file,'rb') as outfile:
            extra_list = np.load(outfile).tolist()

    if with_extra_sv:
        print(len(train_list), len(val_list),len(extra_list))
    else:
        print(len(train_list), len(val_list))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'#args['gpu_id']

    if args['model_type'] == 'token':
        model = CSRNet()
    else:
        model = CSRNet()

    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    criterion = nn.L1Loss(size_average=False).cuda()

    optimizer = torch.optim.Adam(
        [  #
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.1, last_epoch=-1)
    print(args['pre'])

    # args['save_path'] = args['save_path'] + str(args['rdt'])
    print(args['save_path'])
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])

    print(args['best_pred'], args['start_epoch'])
    train_data = pre_data(train_list, args, train=True)
    test_data = pre_data(val_list, args, train=False)
    if with_extra_sv:
        extra_data = pre_data(extra_list,args,train=True)

    for epoch in range(args['start_epoch'], args['epochs']):

        start = time.time()
        train(train_data, model, criterion, optimizer, epoch, args, scheduler)
        #train_extra(extra_data, model, criterion, optimizer, epoch, args, scheduler)

        end1 = time.time()

        if epoch % 1 == 0 and epoch >= 0:
            prec1 = validate(test_data, model, args)
            end2 = time.time()
            is_best = prec1 < args['best_pred']
            args['best_pred'] = min(prec1, args['best_pred'])

            print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']), args['save_path'], end1 - start, end2 - end1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': './save_file/ShanghaiB256_9.459_14.094/model_best.pth',#args['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': args['best_pred'],
                'optimizer': optimizer.state_dict(),
            }, is_best, args['save_path'])


def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

        '''for debug'''
        # if j> 10:
        #     break
    return data_keys


def train(Pre_data, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor(),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))

    model.train()
    end = time.time()

    for i, (fname, img, gt_count) in enumerate(train_loader):

        data_time.update(time.time() - end)

        mask = create_mask()
        mask_p = torch.tensor(mask).type(torch.FloatTensor)
        img_p = img*mask_p
        img_n = img*(1.0-mask_p)
        img = torch.cat((img, img_p, img_n), dim=0)
        img = img.cuda()

        out1 = model(img)
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)

        out_ori = out1[0:12]
        out_mask_p =  out1[12:24]
        out_mask_n = out1[24:36]
        out_sum = (out_mask_p+out_mask_n)

        # print(out1.shape, kpoint.shape)
        loss = criterion(out_ori, gt_count) + 0.5*criterion(out_sum, out_ori)+ 0.5*criterion(out_sum,  gt_count)# + 0.0001*criterion(image_self, img)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1 == 0:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    scheduler.step()

def create_mask(width=256,height=256,mask_size=100,x=None,y=None):
    mask_size_x = random.randint(20, 230)  #(50,200) 100
    mask_size_y = random.randint(20, 230)  #100
    mask = np.ones((height,width))
    mask_x= x if x is not None else random.randint(0,width-mask_size_x) #(0,156) 100
    mask_y= y if y is not None else random.randint(0,height-mask_size_y) #(0,156) 50
    mask[mask_y : mask_y + mask_size_y, mask_x:mask_x + mask_size_x] = 0
    return mask


def train_extra(Pre_data, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor(),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))

    model.train()
    end = time.time()

    for i, (fname, img, gt_count) in enumerate(train_loader):

        data_time.update(time.time() - end)
        img = img.cuda()
        mask = create_mask()
        mask_p = torch.tensor(mask).type(torch.FloatTensor).cuda()


        #out1 = model(img*mask_p)
        #out2 = model(img*(1-mask_p))

        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)

        # print(out1.shape, kpoint.shape)
        loss = criterion(out1, gt_count)# + 0.0001*criterion(image_self, img)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1 == 0:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    scheduler.step()

def validate(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    index = 0

    for i, (fname, img, gt_count) in enumerate(test_loader):

        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            out1 = model(img)
            count = torch.sum(out1).item()

        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % 15 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    nni.report_intermediate_result(mae)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    return mae


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


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
