import argparse
import sys
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../dataset/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/test', help='saving_path')

# Model specifications
parser.add_argument('--use_adaptive_mask', type=bool, default=False, help='use dynamic mask')
parser.add_argument('--pretrained_mask_path', type=str, default=None, help='pretrained model directory')
parser.add_argument('--method', type=str, default='DMDC_9stg', help='method name')
parser.add_argument("--input_setting", type=str, default='Y', help='the input measurement of the network: H, HM or Y DMDC')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--loss", type=bool, default=False, help='use loss2')
parser.add_argument("--input_mask", type=str, default='Phi', help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications
parser.add_argument("--patch_size", type=int, default=256, help="patch size")
parser.add_argument("--stride", type=int, default=112, help="stride")
parser.add_argument("--channel", type=int, default=31, help="stride")
parser.add_argument('--batch_size', type=int, default=2, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=200, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=2000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0003)


opt = parser.parse_args()
template.set_template(opt)

# dataset
opt.data_path = f"{opt.data_root}/Train_Spec/"
opt.data_path_RGB = f"{opt.data_root}/Train_RGB/"
opt.mask_path = opt.data_root
opt.train_split_txt = f"{opt.data_root}/split_txt/train_list.txt"
opt.test_split_txt = f"{opt.data_root}/split_txt/valid_list.txt"




for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False