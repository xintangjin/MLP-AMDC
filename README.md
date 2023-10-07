# MLP-AMDC

# ARAD_1K
## 1. Create Envirement:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  cd AMDC
  pip install -r requirements.txt
  ```
## 2. Data Preparation:

- Download training spectral images ([Google Drive](https://drive.google.com/file/d/1FQBfDd248dCKClR-BpX5V2drSbeyhKcq/view) / [Baidu Disk](https://pan.baidu.com/s/1NisQ6NjGvVhc0iOLH7OFvg), code: `mst1`), training RGB images ([Google Drive](https://drive.google.com/file/d/1A4GUXhVc5k5d_79gNvokEtVPG290qVkd/view) / [Baidu Disk](https://pan.baidu.com/s/1k7aSSL5MMipWYszlFaBLkA)),  validation spectral images ([Google Drive](https://drive.google.com/file/d/12QY8LHab3gzljZc3V6UyHgBee48wh9un/view) / [Baidu Disk](https://pan.baidu.com/s/1CIb5AqLWJxaGilTPtmWl0A)), validation RGB images ([Google Drive](https://drive.google.com/file/d/19vBR_8Il1qcaEZsK42aGfvg5lCuvLh1A/view) / [Baidu Disk](https://pan.baidu.com/s/1YakbXgBgnhNmYoxySmZaGw)), and testing RGB images ([Google Drive](https://drive.google.com/file/d/1A5309Gk7kNFI-ORyADueiPOCMQNTA7r5/view) / [Baidu Disk](https://pan.baidu.com/s/1RXHK64mUfK_GeeoLzqAmeQ)) from the [competition website](https://codalab.lisn.upsaclay.fr/competitions/721#participate-get_data) of NTIRE 2022 Spectral Reconstruction Challenge.

- Place the training spectral images and validation spectral images to `/AMDC/dataset/Train_Spec/`.

- Place the training RGB images and validation RGB images to `/AMDC/dataset/Train_RGB/`.

- Place the testing RGB images  to `/AMDC/dataset/Test_RGB/`.

- Then this repo is collected as the following form:

  ```shell
  |--AMDC
      |--ARAD_1K 
      |--dataset 
          |--Train_Spec
              |--ARAD_1K_0001.mat
              |--ARAD_1K_0002.mat
              ： 
              |--ARAD_1K_0950.mat
		  |--Train_RGB
              |--ARAD_1K_0001.jpg
              |--ARAD_1K_0002.jpg
              ： 
              |--ARAD_1K_0950.jpg
          |--Valid_Spec
              |--ARAD_1K_0901.mat
              |--ARAD_1K_0902.mat
              ： 
              |--ARAD_1K_0950.mat
		  |--Valid_RGB
              |--ARAD_1K_0901.jpg
              |--ARAD_1K_0902.jpg
              ： 
              |--ARAD_1K_0950.jpg
          |--Test_RGB
              |--ARAD_1K_0951.jpg
              |--ARAD_1K_0952.jpg
              ： 
              |--ARAD_1K_1000.jpg
          |--split_txt
              |--train_list.txt
              |--valid_list.txt
          |--mask.mat
  ```
  
## 3. Training
```shell
cd /AMDC/ARAD_1K/train_code/
python train.py --method AMDC_3stg  --batch_size 4  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python train.py --method AMDC_5stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python train.py --method AMDC_7stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python train.py --method AMDC_9stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
```
## 4. Testing
(1)  Download the pretrained model zoo from (will coming soon). 

(2)  Run the following command to test the model on the testing RGB images. 
```shell
cd /AMDC/ARAD_1K/test_code/
python test.py --method AMDC_3stg  --batch_size 4  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python test.py --method AMDC_5stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python test.py --method AMDC_7stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python test.py --method AMDC_9stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
```


# KAIST and CAVE adhering to the TSA-Net
## 1. Data Preparation:
- The repo is collected as the following form:

  ```shell
  |--AMDC
      |--simulation 
      |--datasets 
          |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
		  |--cave_1024_28_RGB
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
          |--cave_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
          |--cave_512_28_RGB
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
		  |--TSA_simu_data
              |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
              |--Truth_RGB
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
              |--mask.mat
  ```
1)Download cave_1024_28 ([One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:
![#f03c15](https://placehold.it/15/f03c15/000000?text=+) 2) Download cave_1024_28_RGB,Truth_RGB ([Baidu Disk](https://pan.baidu.com/s/1GINXZM0nAe-EKlsXN0uqzA?pwd=y183), code: `y183`))

##2.Prepare Pretrained ckpt:

Download pretrained (will comming soon) .

## 3. Training
```shell
cd /AMDC/simulation/train_code/
python train.py --method AMDC_3stg  --batch_size 4  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python train.py --method AMDC_5stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python train.py --method AMDC_7stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python train.py --method AMDC_9stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
```
## 4. Testing
```shell
cd /AMDC/simulation/test_code/
python test.py --method AMDC_3stg  --batch_size 4  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python test.py --method AMDC_5stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python test.py --method AMDC_7stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
python test.py --method AMDC_9stg  --batch_size 2  --outf ./exp/AMDC_3stg/ --data_root ../dataset/  --gpu_id 0
```