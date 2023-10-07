import argparse


parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='1')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/AMDC_9stg/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='AMDC_9stg', help='method name')
parser.add_argument('--use_adaptive_mask', type=bool, default=True, help='use adaptive mask')
parser.add_argument('--pretrained_model_path', type=str, default= None, help='pretrained model directory')
parser.add_argument('--pretrained_mask_path', type=str, default=None, help='pretrained model directory')

parser.add_argument("--input_setting", type=str, default='Y',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi_PhiPhiT',
                    help='the input mask of the network: Phi, Phi_PhiPhiT or None')

opt = parser.parse_args()

opt.mask_path = f"{opt.data_root}/TSA_simu_data/"
opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"
opt.test_path_RGB = f"{opt.data_root}/TSA_simu_data/Truth_RGB/"


for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False