import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from ssim_torch import ssim


def generate_shift_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch**2,1)
    Phi_s_batch[Phi_s_batch==0] = 1
    # print(Phi_batch.shape, Phi_s_batch.shape)
    return Phi_batch, Phi_s_batch

def LoadTraining(path, RGB_path):
    imgs = []
    RGBs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    for i in range(len(scene_list)):
    # for i in range(10):
        scene_path = path + scene_list[i]
        scene_RGB_path = RGB_path + scene_list[i]

        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num<=205:
            if 'mat' not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict['img_expand'] / 65536.
            elif "img" in img_dict:
                img = img_dict['img'] / 65536.
            img = img.astype(np.float32)
            imgs.append(img)
            print('Sence {} is loaded. {}'.format(i, scene_list[i]))

            RGB_dict = sio.loadmat(scene_RGB_path)
            RGB = RGB_dict['RGB']
            RGBs.append(RGB)
            print('Sence {} RGB is loaded. {}'.format(i, scene_list[i]))
    return imgs, RGBs

def LoadTest(path_test, test_path_RGB):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    test_RGB = np.zeros((len(scene_list), 256, 256, 3))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img

        scene_path_RGB = test_path_RGB + scene_list[i]
        RGB = sio.loadmat(scene_path_RGB)['RGB']
        test_RGB[i, :, :, :] = RGB

    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    test_RGB = torch.from_numpy(np.transpose(test_RGB, (0, 3, 1, 2)))
    return test_data, test_RGB

def LoadMeasurement(path_test_meas):
    img = sio.loadmat(path_test_meas)['simulation_test']
    test_data = img
    test_data = torch.from_numpy(test_data)
    return test_data

# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def shuffle_crop(train_data, train_RGB, batch_size, crop_size=256, argument=True):
    if argument:
        gt_batch = []
        gt_rgb = []
        # The first half data use the original data.
        index = np.random.choice(range(len(train_data)), batch_size//2)
        processed_data = np.zeros((batch_size//2, crop_size, crop_size, 28), dtype=np.float32)
        processed_rgb = np.zeros((batch_size // 2, crop_size, crop_size, 3), dtype=np.float32)
        for i in range(batch_size//2):
            img = train_data[index[i]]
            img_rgb = train_RGB[index[i]]
            h, w, _ = img.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
            processed_rgb[i, :, :, :] = img_rgb[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        processed_rgb = torch.from_numpy(np.transpose(processed_rgb, (0, 3, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            data, rgb = arguement_1(processed_data[i], processed_rgb[i])
            gt_batch.append(data)
            gt_rgb.append(rgb)

        # The other half data use splicing.
        processed_data = np.zeros((4, 128, 128, 28), dtype=np.float32)
        processed_rgb = np.zeros((4, 128, 128, 3), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
                processed_rgb[j] = train_RGB[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]

            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()  # [4,28,128,128]
            gt_rgb_2 = torch.from_numpy(np.transpose(processed_rgb, (0, 3, 1, 2))).cuda()  # [4,3,128,128]
            data2, rgb2 = arguement_2(gt_batch_2, gt_rgb_2)
            gt_batch.append(data2)
            gt_rgb.append(rgb2)
        gt_batch = torch.stack(gt_batch, dim=0)
        gt_rgb = torch.stack(gt_rgb, dim=0)
        return gt_batch, gt_rgb
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        processed_rgb = np.zeros((batch_size, crop_size, crop_size, 3), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
            processed_rgb[i, :, :, :] = train_RGB[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        gt_rgb = torch.from_numpy(np.transpose(processed_rgb, (0, 3, 1, 2)))
        return gt_batch, gt_rgb

def arguement_1(data, rgb):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        data = torch.rot90(data, dims=(1, 2))
        rgb = torch.rot90(rgb, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        data = torch.flip(data, dims=(2,))
        rgb = torch.flip(rgb, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        data = torch.flip(data, dims=(1,))
        rgb = torch.flip(rgb, dims=(1,))
    return data, rgb

def arguement_2(data, rgb):
    c, h, w = data.shape[1],256,256
    c2 = rgb.shape[1]
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(c,h,w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = data[0]
    output_img[:, :divid_point_h, divid_point_w:] = data[1]
    output_img[:, divid_point_h:, :divid_point_w] = data[2]
    output_img[:, divid_point_h:, divid_point_w:] = data[3]

    output_rgb = torch.zeros(c2,h,w).cuda()
    output_rgb[:, :divid_point_h, :divid_point_w] = rgb[0]
    output_rgb[:, :divid_point_h, divid_point_w:] = rgb[1]
    output_rgb[:, divid_point_h:, :divid_point_w] = rgb[2]
    output_rgb[:, divid_point_h:, divid_point_w:] = rgb[3]
    return output_img, output_rgb


def gen_meas_torch(data_batch, mask3d_batch, Y2H=False, mul_mask=True):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    meas = shift_back(meas)
    return meas

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def load_mask(mask_path):
    """ load mask
    output [h, w, 28]"""
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    # mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 1))
    mask = torch.from_numpy(mask)
    return mask

def mask2batch(mask3d, batch_size=1):
    """ input[h, w, 28]
    output [batch_size, h, w, 28]"""
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch


def shift_mask(mask3d, size=256):
    """
    input[b, 28, h, w]
    output[b, 28,h, w+2d]
    """
    b,_,w,h = mask3d.shape
    mask3d_shift = torch.zeros((b, 28, size, size + (28 - 1) * 2)).cuda().float()
    mask3d_shift[:,:, :, 0:size] = mask3d
    for t in range(28):
        mask3d_shift[:, t, :,:] = torch.roll(mask3d_shift[:, t, :, :], 2 * t, dims=2)
    return mask3d_shift

def sum_shift_mask(mask3d_shift):
    """
    input[b, 28, h, w+2d]
    output[b, h, w+2d]
    """
    mask_3d_shift_s = torch.sum(mask3d_shift ** 2, dim=1, keepdim=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    return mask_3d_shift_s


def init_meas(gt, mask, input_setting):
    if input_setting == 'H':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':  #
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    return input_meas


def scend_gen_meas_torch(data_batch, mask3d_batch):
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    meas = shift_back(meas)
    return meas



def second_init_meas(gt, mask):
    meas = scend_gen_meas_torch(gt, mask)
    return meas

def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def checkpoint_mask(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_mask_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint_mask saved to {}".format(model_out_path))