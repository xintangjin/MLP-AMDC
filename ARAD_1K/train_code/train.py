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

mask3d_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
mask3d_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, opt.batch_size)
# dataset
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)


if opt.use_adaptive_mask == True:
    model_mask = mask_generator(mask_base, opt.pretrained_mask_path).cuda()
    # model_mask.load_state_dict(torch.load('/model_mask_epoch_146.pth'))

model = model_generator(opt.method, opt.pretrained_model_path).cuda()
# model.load_state_dict(torch.load('/model_epoch_144.pth'))


# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.use_adaptive_mask == True:
    optimizer_mask = torch.optim.Adam(model_mask.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    if opt.use_adaptive_mask == True:
        scheduler_mask = torch.optim.lr_scheduler.MultiStepLR(optimizer_mask, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
    if opt.use_adaptive_mask == True:
        scheduler_mask = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mask, opt.max_epoch, eta_min=1e-6)


criterion = losses.CharbonnierLoss().cuda()

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_ssim = Loss_SSIM()
criterion_mrae.cuda()
criterion_rmse.cuda()
criterion_psnr.cuda()
criterion_ssim.cuda()

def test(val_loader, model, input_mask_test, mask3d_batch_test):
    if opt.use_adaptive_mask == True:
        model_mask.eval()
    model.eval()

    mrae = AverageMeter()
    rmse = AverageMeter()
    psnr = AverageMeter()
    SSIM = AverageMeter()

    begin = time.time()
    for i, (test_RGB, test_gt) in enumerate(val_loader):
        test_RGB = test_RGB.cuda().float()
        test_gt = test_gt.cuda().float()

        if opt.method in ['AMDC_3stg', 'AMDC_1stg', 'AMDC_5stg',  'AMDC_7stg', 'AMDC_9stg', 'OLU']:

            if opt.use_adaptive_mask == True:
                with torch.no_grad():
                    mask_out = model_mask(test_gt.shape[0])
            else:
                mask_out = torch.repeat_interleave(mask_base.unsqueeze(0), test_gt.shape[0], dim=0)

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
                if opt.input_setting == 'AMDC':
                    model_out = model(input_meas, test_RGB, input_mask_test)
                else:
                    model_out = model(input_meas, input_mask_test)
                loss_mrae = criterion_mrae(model_out[:, :, :, :], test_gt[:, :, :, :])
                loss_rmse = criterion_rmse(model_out[:, :, :, :], test_gt[:, :, :, :])
                loss_psnr = psnr_loss(model_out[:, :, :, :].clamp(0., 1.).data.cpu().numpy(), test_gt[:, :, :, :].clamp(0., 1.).data.cpu().numpy())
                loss_ssim = criterion_ssim(model_out[:, :, :, :], test_gt[:, :, :, :])

        elif opt.method in ['cst_s', 'cst_m', 'cst_l']:
            if test_gt.shape[0] != mask3d_batch_test.shape[0]:
                mask3d_batch_test = mask3d_batch_test[:test_gt.shape[0], :, :, :]
                input_mask1 = input_mask_test
                input_mask1 = input_mask1[:test_gt.shape[0], :, :]
                input_mask_test = input_mask1
            input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
            model_out, _ = model(input_meas, input_mask_test)
            loss_mrae = criterion_mrae(model_out[:, :, :, :], test_gt[:, :, :, :])
            loss_rmse = criterion_rmse(model_out[:, :, :, :], test_gt[:, :, :, :])
            loss_psnr = psnr_loss(model_out[:, :, :, :].clamp(0., 1.).data.cpu().numpy(),
                                  test_gt[:, :, :, :].clamp(0., 1.).data.cpu().numpy())
            loss_ssim = criterion_ssim(model_out[:, :, :, :], test_gt[:, :, :, :])

        else:
            if opt.method in ['mst_plus_plus', 'lambda_net']:
                if test_gt.shape[0] != mask3d_batch_test.shape[0]:
                    mask3d_batch_test = mask3d_batch_test[:test_gt.shape[0],:,:,:]
                    input_mask1 = input_mask_test
                    input_mask1 = input_mask1[:test_gt.shape[0], :, :]
                    input_mask_test = input_mask1
            elif opt.method in ['tsa_net']:
                if test_gt.shape[0] != mask3d_batch_test.shape[0]:
                    mask3d_batch_test = mask3d_batch_test[:test_gt.shape[0], :, :, :]

            else:
                if test_gt.shape[0] != mask3d_batch_test.shape[0]:
                    mask3d_batch_test = mask3d_batch_test[:test_gt.shape[0],:,:,:]
                    mask3d_batch1, input_mask1 = input_mask_test
                    mask3d_batch1 = mask3d_batch1[:test_gt.shape[0],:,:,:]
                    input_mask1 = input_mask1[:test_gt.shape[0], :, :]
                    input_mask_test = mask3d_batch1, input_mask1

            input_meas = init_meas(test_gt, mask3d_batch_test, opt.input_setting)
            model_out = model(input_meas, input_mask_test)
            loss_mrae = criterion_mrae(model_out[:, :, :, :], test_gt[:, :, :, :])
            loss_rmse = criterion_rmse(model_out[:, :, :, :], test_gt[:, :, :, :])
            loss_psnr = psnr_loss(model_out[:, :, :, :].clamp(0., 1.).data.cpu().numpy(),
                                  test_gt[:, :, :, :].clamp(0., 1.).data.cpu().numpy())
            loss_ssim = criterion_ssim(model_out[:, :, :, :], test_gt[:, :, :, :])

        mrae.update(loss_mrae.data)
        rmse.update(loss_rmse.data)
        psnr.update(loss_psnr)
        SSIM.update(loss_ssim.data)
    end = time.time()

    return mrae.avg, rmse.avg, psnr.avg, SSIM.avg, end-begin

def main(mask3d_batch_train, input_mask_train, input_mask_test, mask3d_batch_test):
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    iteration = 0
    record_mrae_loss = 1000
    begin = time.time()
    for epoch in range(1, opt.max_epoch + 1):
        if opt.use_adaptive_mask == True:
            model_mask.train()
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        for i, (gt_rgb, gt_batch) in enumerate(train_loader):
            gt = Variable(gt_batch).cuda().float()  # [b, 31, w, h]
            rgb = Variable(gt_rgb).cuda().float()  # [b, 3, w, h]
            lr = optimizer.param_groups[0]['lr']

            if opt.method in ['DMDC_3stg', 'DMDC_1stg', 'DMDC_5stg',  'DMDC_7stg', 'DMDC_9stg', 'OLU']:
                optimizer.zero_grad()
                if opt.use_adaptive_mask == True:
                    optimizer_mask.zero_grad()
                    mask_out = model_mask(opt.batch_size) #[b, w, h]
                    name = model_path + 'mask.mat'
                    scio.savemat(name, {'mask': mask_out.detach().cpu().numpy()})
                else:
                    mask_out = torch.repeat_interleave(mask_base.unsqueeze(0), opt.batch_size, dim=0)

                mask3d = torch.repeat_interleave(mask_out.unsqueeze(1), opt.channel, dim=1)  # [b, 31, w, h]
                Phi = shift_mask(mask3d)
                Phi_s = sum_shift_mask(Phi)

                if opt.input_setting == 'DMDC':
                    Phi_s = shift_back(Phi_s)
                    input_mask_train = (mask3d, Phi_s)
                else:
                    input_mask_train = (Phi, Phi_s)

                input_meas = init_meas(gt, mask3d, opt.input_setting)

                if opt.input_setting == 'DMDC':
                    model_out = model(input_meas, rgb, input_mask_train)
                else:
                    model_out = model(input_meas, input_mask_train)
                output_meas = init_meas(model_out, mask3d, opt.input_setting)

                if opt.loss:
                    if opt.input_setting == 'DMDC':
                        loss = criterion(model_out, gt) + 0.01 * criterion(output_meas, input_meas)
                    else:
                        loss = criterion(model_out, gt) + 0.2 * criterion(output_meas, input_meas)
                else:
                    loss = criterion(model_out, gt)

            elif opt.method in ['cst_s', 'cst_m', 'cst_l']:
                optimizer.zero_grad()
                input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)
                model_out, diff_pred = model(input_meas, input_mask_train)
                loss = criterion(model_out, gt)
                diff_gt = torch.mean(torch.abs(model_out.detach() - gt),dim=1, keepdim=True)  # [b,1,h,w]
                loss_sparsity = F.mse_loss(diff_gt, diff_pred)
                loss = loss + 2 * loss_sparsity
            else:
                optimizer.zero_grad()
                input_meas = init_meas(gt, mask3d_batch_train, opt.input_setting)
                model_out = model(input_meas, input_mask_train)
                loss = criterion(model_out, gt)

            loss.backward()
            losses.update(loss.data)
            optimizer.step()
            if opt.use_adaptive_mask == True:
                optimizer_mask.step()
            iteration = iteration+1
            if iteration % 500 == 0:
                end = time.time()-begin
                begin = time.time()
                logger.info(
                    '===> iteration = {:d} | Epoch {}：  learning rate : {:.9f} , train_losses.avg={:.9f} , time: {:.4f}'
                    .format(iteration, epoch, lr, losses.avg, end))
        #test
        scheduler.step()
        if opt.use_adaptive_mask == True:
            scheduler_mask.step()
        mrae_mean, rmse_mean, psnr_mean, ssim_mean, recon_time= test(val_loader, model,input_mask_test, mask3d_batch_test)
        logger.info(
            '===> Epoch {}： testing psnr = {:.2f}, ssim = {:.4f}, mrae: {:.9f}, rmse: {:.9f} , reconstruct_time: {:.4f}'
            .format(epoch, psnr_mean, ssim_mean, mrae_mean, rmse_mean, recon_time))
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            checkpoint(model, epoch, model_path, logger)
            if opt.use_adaptive_mask == True:
                checkpoint_mask(model_mask, epoch, model_path, logger)
        elif mrae_mean < record_mrae_loss:
            record_mrae_loss = mrae_mean
            checkpoint(model, epoch, model_path, logger)
            if opt.use_adaptive_mask == True:
                checkpoint_mask(model_mask, epoch, model_path, logger)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main(mask3d_batch_train, input_mask_train,input_mask_test, mask3d_batch_test)

