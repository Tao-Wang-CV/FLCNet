import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
from model import FLCNet
from utils.data_val import test_dataset

model_name = 'FLCNet'
model_pth = 'best_FLCNet.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./saved_models/{}/{}'.format(model_name,model_pth))
opt = parser.parse_args()

for _data_name in ['CAMO_TestingDataset', 'CHAMELEON_TestingDataset', 'COD10K_TestingDataset', 'NC4K_TestingDataset']:
    data_path = './dataset/testdata/{}/'.format(_data_name ,_data_name) #测试集地址
    save_path = './pre/{}/{}/'.format(model_name ,_data_name.replace('_TestingDataset',''))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = FLCNet()

    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Image/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        d1, d2, d3, d4 = model(image)

        res = d1
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))

        cv2.imwrite(save_path+name,res*255)
