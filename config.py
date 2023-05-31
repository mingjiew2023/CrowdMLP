import argparse

parser = argparse.ArgumentParser(description='TransCrowd')

# Data specifications
parser.add_argument('--dataset', type=str, default='ShanghaiA256',#'ShanghaiB256',#'UCF_QNRF',
                    help='choice train dataset')

parser.add_argument('--save_path', type=str, default='./save_file/SHA_cls_ada_new_clip',#./save_file/QNRF_94.130_170.324_1024_768',
                    help='save checkpoint directory')

parser.add_argument('--workers', type=int, default=16,
                    help='load data workers')
parser.add_argument('--print_freq', type=int, default=200,
                    help='print frequency')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch for training')

# Model specifications
parser.add_argument('--test_dataset', type=str, default=None,
                    help='choice train dataset')
parser.add_argument('--pre', type=str, default='./save_file/SHA_cls_ada_new_clip/checkpoint.pth',#'./save_file/QNRF_94.130_170.324_1024_768/model_best.pth',
                    help='pre-trained model directory')
# parser.add_argument('--pre', type=str, default='./save_file/A_baseline_4/model_best_66.1.pth',
#                     help='pre-trained model directory')B


# Optimization specifications
parser.add_argument('--batch_size', type=int, default=24,#36,
                    help='input batch size for training')
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4,
                    help='weight decay')
parser.add_argument('--momentum', type=float, default=0.95,
                    help='momentum')
parser.add_argument('--epochs', type=int, default=20000,
                    help='number of epochs to train')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--best_pred', type=int, default=1e5,
                    help='best pred')
parser.add_argument('--gpu_id', type=str, default='1',
                    help='gpu id')

# nni config
parser.add_argument('--lr', type=float, default=1e-5,#1e-5,
                    help='learning rate')
parser.add_argument('--model_type', type=str, default='token',
                    help='model type')

args = parser.parse_args()
return_args = parser.parse_args()
