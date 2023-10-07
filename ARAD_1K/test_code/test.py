import os
from option import opt
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
from architecture import *
from utils import *
import torch
import scipy.io as scio
import time
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import datetime
from tqdm import tqdm
import torch.nn as nn
import losses
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

mask_base = load_mask(opt.mask_path).cuda() #[w, h]
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
# dataset
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))

# saving path
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if opt.use_adaptive_mask == True:
    model_mask = mask_generator(mask_base, opt.pretrained_mask_path).cuda()
    model_mask.load_state_dict(torch.load('/model_mask_epoch_213.pth'))

model = model_generator(opt.method, opt.pretrained_model_path).cuda()
model.load_state_dict(torch.load('/model_epoch_213.pth'))


def test(val_loader, model, input_mask_test, mask3d_batch_test):
    criterion_mrae = Loss_MRAE()
    criterion_rmse = Loss_RMSE()
    criterion_psnr = Loss_PSNR()
    criterion_ssim = Loss_SSIM()

    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_ssim.cuda()

    if opt.use_adaptive_mask == True:
        model_mask.eval()
    model.eval()

    mrae = AverageMeter()
    rmse = AverageMeter()
    psnr = AverageMeter()
    SSIM = AverageMeter()
    TIME = AverageMeter()

    PRED = []
    TRUTH = []
    RGB = []
    MASK = []

    for i, (test_RGB, test_gt) in enumerate(val_loader):
        test_RGB = test_RGB.cuda().float()
        test_gt = test_gt.cuda().float()

        if opt.method in ['AMDC_3stg', 'AMDC_1stg', 'AMDC_5stg', 'AMDC_7stg', 'AMDC_9stg', 'OLU']:
            if opt.use_adaptive_mask == True:
                with torch.no_grad():
                    begin = time.time()
                    mask_out = model_mask(test_RGB)
                    time1 = time.time() - begin
            else:
                mask_out = torch.repeat_interleave(mask_base.unsqueeze(0), opt.batch_size, dim=0)

            mask3d = torch.repeat_interleave(mask_out.unsqueeze(1), 31, dim=1)
            Phi = shift_mask(mask3d)
            Phi_s = sum_shift_mask(Phi)
            if opt.input_setting == 'AMDC':
                Phi_s = shift_back(Phi_s)
                input_mask_test = (mask3d, Phi_s)
            else:
                input_mask_test = (Phi, Phi_s)

            input_meas = init_meas(test_gt, mask3d, opt.input_setting)
            with torch.no_grad():
                begin = time.time()
                if opt.input_setting == 'AMDC':
                    model_out = model(input_meas, test_RGB, input_mask_test)
                else:
                    model_out = model(input_meas, input_mask_test)
                time_all = time.time() - begin + time1
                loss_mrae = criterion_mrae(model_out[:, :, :, :], test_gt[:, :, :, :])
                loss_rmse = criterion_rmse(model_out[:, :, :, :], test_gt[:, :, :, :])
                loss_psnr = psnr_loss(model_out[:, :, :, :].clamp(0., 1.).data.cpu().numpy(), test_gt[:, :, :, :].clamp(0., 1.).data.cpu().numpy())
                loss_ssim = criterion_ssim(model_out[:, :, :, :], test_gt[:, :, :, :])

        elif opt.method in ['cst_s', 'cst_m', 'cst_l']:

            input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
            begin = time.time()
            model_out, _ = model(input_meas, input_mask_test)
            time_all = time.time() - begin
            loss_mrae = criterion_mrae(model_out[:, :, :, :], test_gt[:, :, :, :])
            loss_rmse = criterion_rmse(model_out[:, :, :, :], test_gt[:, :, :, :])
            loss_psnr = psnr_loss(model_out[:, :, :, :].clamp(0., 1.).data.cpu().numpy(),
                                  test_gt[:, :, :, :].clamp(0., 1.).data.cpu().numpy())
            loss_ssim = criterion_ssim(model_out[:, :, :, :], test_gt[:, :, :, :])


        else:
            if opt.method in ['mst_plus_plus', 'lambda_net']:
                if test_gt.shape[0] != mask3d_batch_test.shape[0]:
                    mask3d_batch_test = mask3d_batch_test[:test_gt.shape[0], :, :, :]
                    input_mask1 = input_mask_test
                    input_mask1 = input_mask1[:test_gt.shape[0], :, :]
                    input_mask_test = input_mask1

            elif opt.method in ['tsa_net']:
                if test_gt.shape[0] != mask3d_batch_test.shape[0]:
                    mask3d_batch_test = mask3d_batch_test[:test_gt.shape[0], :, :, :]

            else:
                if test_gt.shape[0] != mask3d_batch_test.shape[0]:
                    mask3d_batch_test = mask3d_batch_test[:test_gt.shape[0], :, :, :]
                    mask3d_batch1, input_mask1 = input_mask_test
                    mask3d_batch1 = mask3d_batch1[:test_gt.shape[0], :, :, :]
                    input_mask1 = input_mask1[:test_gt.shape[0], :, :]
                    input_mask_test = mask3d_batch1, input_mask1

            input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
            begin = time.time()
            model_out = model(input_meas, input_mask_test)
            time_all = time.time() - begin
            loss_mrae = criterion_mrae(model_out[:, :, :, :], test_gt[:, :, :, :])
            loss_rmse = criterion_rmse(model_out[:, :, :, :], test_gt[:, :, :, :])
            loss_psnr = psnr_loss(model_out[:, :, :, :].clamp(0., 1.).data.cpu().numpy(),
                                  test_gt[:, :, :, :].clamp(0., 1.).data.cpu().numpy())
            loss_ssim = criterion_ssim(model_out[:, :, :, :], test_gt[:, :, :, :])

        mrae.update(loss_mrae.data)
        rmse.update(loss_rmse.data)
        psnr.update(loss_psnr)
        SSIM.update(loss_ssim.data)
        TIME.update(time_all)

        pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        rgb = np.transpose(test_RGB.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        mask_out = (mask_out.cpu().numpy()).astype(np.float32)
        PRED.append(pred)
        TRUTH.append(truth)
        RGB.append(rgb)
        MASK.append(mask_out)
    end_time = TIME.sum

    return PRED, TRUTH, RGB, MASK, mrae.avg, rmse.avg, psnr.avg, SSIM.avg, end_time

def main(input_mask_test,mask3d_batch_test):
    for epoch in range(3):
        val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        #test
        pred, truth, rgb, mask, mrae_mean, rmse_mean, psnr_mean, ssim_mean, recon_time= test(val_loader, model,input_mask_test, mask3d_batch_test)
        print(
            '===> Epoch {}： testing psnr = {:.2f}, ssim = {:.4f}, mrae: {:.9f}, rmse: {:.9f} , reconstruct_time: {:.4f}'
            .format(epoch, psnr_mean, ssim_mean, mrae_mean, rmse_mean, recon_time))

        name = opt.outf + '/Test_result.mat'
        print(f'Save reconstructed HSIs as {name}.')
        scio.savemat(name, {'truth': truth, 'pred': pred, 'rgb':rgb, 'mask':mask})

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main(input_mask_test, mask3d_batch_test)

