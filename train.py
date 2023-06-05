import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from utils.data_val import get_loader

import numpy as np
import glob


from model import FLCNet

import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_iou_loss(pred,target):

	bce_out = bce_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out+ iou_out

	return loss

def muti_bce_loss_fusion( d1, d2, d3 ,d4, labels_v):

    loss1 = bce_iou_loss(d1, labels_v)
    loss2 = bce_iou_loss(d2, labels_v)
    loss3 = bce_iou_loss(d3, labels_v)
    loss4 = bce_iou_loss(d4, labels_v)
    loss = loss1 +loss2 + loss3+ loss4

    #print("l1: %3f, l2: %3f, l3: %3f,l4: %4f,\n"%(loss1.item(),loss2.item(),loss3.item(),loss4.item()))
    return loss1 ,loss

# ------- 2. set the directory of training dataset --------

data_dir = './dataset/traindata/'  #训练集地址
tra_image_dir = 'COD10K/Image/'
tra_label_dir = 'COD10K/GT/'
image_ext = '.jpg'
label_ext = '.png'

epoch_num = 100
batch_size_train = 14
lr = 0.0001
size_of_image = 352
train_num = 0
val_num = 0

model_name = 'FLCNet'

model_dir = "./saved_models/{}/".format(model_name)
tensorboard_dir = './tensorboardx/{}/'.format(model_name)

if not os.path.exists(model_dir ):
    os.makedirs(model_dir)
if not os.path.exists(tensorboard_dir ):
    os.makedirs(tensorboard_dir)

writer = SummaryWriter(tensorboard_dir)


tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split("/")[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_loader = get_loader(image_root=data_dir + tra_image_dir  ,
                          gt_root=data_dir + tra_label_dir  ,
                              batchsize=batch_size_train,
                              trainsize=size_of_image,
                              shuffle=True,
                              num_workers=8)
total_step = len(train_loader)


# ------- 3. define model --------
# define the net
torch.backends.cudnn.enabled=True
net = FLCNet()
net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
Number = 0

for epoch in range(0, epoch_num):

    for i, (images, gts) in enumerate(train_loader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs = images  # .cuda()
        labels = gts  # .cuda()

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        # if torch.cuda.is_available():
        inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d1, d2, d3,d4 = net(inputs_v)
        loss1, loss = muti_bce_loss_fusion(d1, d2, d3,d4, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.item()
        running_tar_loss += loss1.item()

        # del temporary outputs and loss
        del  d1, d2, d3, d4,loss1, loss


        if i % 20 == 0 or i+1 == total_step or i == 1:
             print('Epoch [{:d}/{:d}],Step [{:d}/{:d}],ite: {:d},lr:{:.6f}]Total_loss: {:.5f} Loss1: {:0.5f}'.\
                format(epoch+1, epoch_num, i+1, total_step,ite_num, optimizer.param_groups[0]['lr'], running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num  % 200 == 0 or ite_num  == len(tra_img_name_list) or ite_num  == 1:
            writer.add_scalar('Loss', running_loss / ite_num4val, global_step=ite_num)


        if ite_num % 500 == 0:  # save model every 2000 iterations

            torch.save(net.state_dict(), model_dir + "FLCNet_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0


print('-------------Congratulations! Training Done!!!-------------')
