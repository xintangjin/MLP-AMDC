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
import datetime
from tqdm import tqdm
import torch.nn as nn
import losses
import torch.nn.functional as F
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

mask_base = load_mask(opt.mask_path).cuda() #[w, h]

# dataset
train_set, train_RGB = LoadTraining(opt.data_path, opt.data_path_RGB) #[[1024,1024,28],[]]  [[1024,1024,3],[]]
test_data, test_RGB = LoadTest(opt.test_path, opt.test_path_RGB) #[10,28,256,256] [10,3,256,256]
test_RGB = test_RGB.cuda().float()
# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# model
if opt.use_adaptive_mask == True:
    model_mask = mask_generator(mask_base, opt.pretrained_mask_path).cuda()
    # model_mask.load_state_dict(torch.load('/model_mask_epoch_10.pth'))

model = model_generator(opt.method, opt.pretrained_model_path).cuda()
# model.load_state_dict(torch.load('/model_epoch_10.pth'))

# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
optimizer_mask = torch.optim.Adam(model_mask.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    scheduler_mask = torch.optim.lr_scheduler.MultiStepLR(optimizer_mask, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
    scheduler_mask = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mask, opt.max_epoch, eta_min=1e-6)

criterion = losses.CharbonnierLoss().cuda()


def train(epoch, logger):
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_loss = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    train_tqdm = tqdm(range(batch_num))
    for i in train_tqdm:
    # for i in range(batch_num):
        gt_batch, gt_rgb = shuffle_crop(train_set, train_RGB, opt.batch_size)

        gt = Variable(gt_batch).cuda().float() #[b, 28, w, h]
        rgb = Variable(gt_rgb).cuda().float() #[b, 3, w, h]

        optimizer.zero_grad()
        optimizer_mask.zero_grad()

        if opt.use_adaptive_mask == True:
            mask_out = model_mask(rgb) #[b, w, h]
        else:
            mask_out = torch.repeat_interleave(mask_base.unsqueeze(0), opt.batch_size, dim=0)

        mask3d = torch.repeat_interleave(mask_out.unsqueeze(1), 28, dim=1) #[b, 28, w, h]
        Phi = shift_mask(mask3d)
        Phi_s = sum_shift_mask(Phi)
        Phi_s = shift_back(Phi_s)
        input_mask_train = (mask3d, Phi_s)

        input_meas = init_meas(gt, mask3d, opt.input_setting)
        # with torch.no_grad():
        model_out = model(input_meas, rgb, input_mask_train)

        output_meas = init_meas(model_out, mask3d, opt.input_setting)


        loss1 = criterion(model_out, gt)
        loss2 = criterion(output_meas, input_meas)
        loss = loss1 + 0.02*loss2

        epoch_loss1 += loss1.data
        epoch_loss2 += loss2.data
        epoch_loss += loss.data

        loss.backward()
        optimizer.step()
        optimizer_mask.step()

    end = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss1: {:.6f} Avg. Loss2: {:.6f} Avg. Loss: {:.6f} time: {:.2f} lr: {:.6f}".
                format(epoch, epoch_loss1 / batch_num, epoch_loss2 / batch_num, epoch_loss / batch_num, (end - begin), optimizer.param_groups[0]["lr"]))
    return 0

def test(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    model_mask.eval()
    model.eval()
    begin = time.time()

    with torch.no_grad():
        mask_out = model_mask(test_RGB)
    end = time.time()
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
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, mask_time: {:.4f} , reconstruct_time: {:.4f}'
                .format(epoch, psnr_mean, ssim_mean,(end - begin),(end1 - begin1)))
    model_mask.train()
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    psnr_max = 0
    for epoch in range(1, opt.max_epoch + 1):
        train(epoch, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)
        scheduler.step()
        scheduler_mask.step()
        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 28:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                checkpoint(model, epoch, model_path, logger)
                checkpoint_mask(model_mask, epoch, model_path, logger)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()


