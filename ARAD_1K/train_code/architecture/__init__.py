import torch
from .OLU import OLU
from .AMDC import AMDC
from .Adaptive_mask import MsakNet

from .MST import MST
from .GAP_Net import GAP_net
from .ADMM_Net import ADMM_net
from .TSA_Net import TSA_Net
from .MST_Plus_Plus import MST_Plus_Plus
from .Lambda_Net import Lambda_Net
from .CST import CST
from .DAUHST import DAUHST

def model_generator(method, pretrained_model_path=None):
    if method.find('OLU') >= 0:
        num_iterations = int(method.split('_')[1][0])
        model = OLU(num_iterations=num_iterations).cuda()
    elif method.find('AMDC') >= 0:
        num_iterations = int(method.split('_')[1][0])
        model = AMDC(num_iterations=num_iterations).cuda()
    elif method == 'mst_s':
        model = MST(dim=31, stage=2, num_blocks=[2, 2, 2]).cuda()
    elif method == 'mst_m':
        model = MST(dim=31, stage=2, num_blocks=[2, 4, 4]).cuda()
    elif method == 'mst_l':
        model = MST(dim=31, stage=2, num_blocks=[4, 7, 5]).cuda()
    elif method == 'gap_net':
        model = GAP_net().cuda()
    elif method == 'admm_net':
        model = ADMM_net().cuda()
    elif method == 'tsa_net':
        model = TSA_Net().cuda()
    elif method == 'mst_plus_plus':
        model = MST_Plus_Plus(in_channels=31, out_channels=31, n_feat=31, stage=3).cuda()
    elif method == 'lambda_net':
        model = Lambda_Net(out_ch=31).cuda()
    elif method == 'cst_s':
        model = CST(num_blocks=[1, 1, 2], sparse=True).cuda()
    elif method == 'cst_m':
        model = CST(num_blocks=[2, 2, 2], sparse=True).cuda()
    elif method == 'cst_l':
        model = CST(num_blocks=[2, 4, 6], sparse=True).cuda()
    elif method == 'cst_l_plus':
        model = CST(num_blocks=[2, 4, 6], sparse=False).cuda()
    elif 'dauhst' in method:
        num_iterations = int(method.split('_')[1][0])
        model = DAUHST(num_iterations=num_iterations).cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)
    return model

def mask_generator(mask_base, pretrained_mask_path):
    model = MsakNet(mask_base).cuda()
    if pretrained_mask_path is not None:
        print(f'load model from {pretrained_mask_path}')
        checkpoint = torch.load(pretrained_mask_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)
    return model