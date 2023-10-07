from architecture import *
from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
from option import opt
from utils import *
import time

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# Intialize mask
mask_base = load_mask(opt.mask_path).cuda() #[w, h]
test_data, test_RGB = LoadTest(opt.test_path, opt.test_path_RGB) #[10,28,256,256] [10,3,256,256]
test_RGB = test_RGB.cuda().float()

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if opt.use_dynamic_mask == True:
    model_mask = mask_generator(mask_base, opt.pretrained_mask_path).cuda()
    model_mask.load_state_dict(torch.load('/model_mask_epoch_191.pth'))

model = model_generator(opt.method, opt.pretrained_model_path).cuda()

model.load_state_dict(torch.load('/model_epoch_191.pth'))


def test():
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    model_mask.eval()
    model.eval()
    begin = time.time()

    with torch.no_grad():
        mask_out = model_mask(test_RGB)
    end = time.time()
    mask = mask_out[1,:,:]
    mask_out = torch.repeat_interleave(mask.unsqueeze(0), 10, dim=0)
    mask3d = torch.repeat_interleave(mask_out.unsqueeze(1), 28, dim=1)
    Phi = shift_mask(mask3d)
    Phi_s = sum_shift_mask(Phi)
    Phi_s = shift_back(Phi_s)
    input_mask_test = (mask3d, Phi_s)

    input_meas = init_meas(test_gt, mask3d, opt.input_setting)
    begin1 = time.time()
    with torch.no_grad():
        model_out = model(input_meas, test_RGB, input_mask_test)
    end1 = time.time()
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    mask = mask_out.cpu().numpy().astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    print('===> testing psnr = {:.2f}, ssim = {:.3f}, mask_time: {:.4f} , reconstruct_time: {:.4f}'
                .format(psnr_mean, ssim_mean,(end - begin),(end1 - begin1)))
    model_mask.train()
    model.train()
    return pred, truth, mask, psnr_list, ssim_list, psnr_mean, ssim_mean


def main():
    (pred, truth, mask, psnr_all, ssim_all, psnr_mean, ssim_mean) = test()
    name = opt.outf + 'Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred, 'mask': mask})

if __name__ == '__main__':
    main()