import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Res2Net_v1b import res2net50_v1b_26w_4s

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class RFB_modified1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified1, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            #BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            #BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(2*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

        self.convx02 = BasicConv2d(in_channel, out_channel, 1)
        self.convx2 = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.convx21 = BasicConv2d(in_channel, out_channel, 3, padding=3, dilation=3)
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        # self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pooling = nn.AdaptiveAvgPool2d((None, None))
        self.conv2 = BasicConv2d(2 * out_channel, 2 * out_channel, 1)
        self.conv3 = BasicConv2d(2 * out_channel, out_channel, 1)

        self.conv_upsample2 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * out_channel, 2 * out_channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * out_channel, 2 * out_channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * out_channel, 3 * out_channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * out_channel, 3 * out_channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * out_channel, 1, 1)
        self.conv6 = nn.Conv2d(3 * out_channel, out_channel, 1)


    def forward(self, x):
        #x0 = self.branch0(x)
        #x1 = self.branch1(x)
        #x_cat = self.conv_cat(torch.cat((x0, x1), 1))
        x_c = self.convx2(x)
        x_cat = self.convx21(x)


        x2_1 = x_cat * x_c
        x3_1 = x_cat + x2_1
        x3_2 = x_c + x2_1
        x4_1 = self.conv1(x3_1)
        x4_2 = self.conv1(x3_2)
        x5_1 = torch.cat((x4_1, x4_2), 1)
        #x5_1_avg_pooling = self.avg_pooling(x5_1)
        #x5_1_avg_pooling = self.conv2(x5_1_avg_pooling)
        tgt_size = x5_1.shape[2:]

        l = F.adaptive_max_pool2d(x5_1, tgt_size) + F.adaptive_avg_pool2d(x5_1, tgt_size)
        x6_1 = self.conv2(l)
        y = self.conv3(x6_1)

        return y

class fuse(nn.Module):
    def __init__(self, channel):
        super(fuse, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(2*channel, channel, 3, padding=1)
        self.conv5 = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2):
        x1_1 = self.upsample(x1)
        x2_1 = torch.cat((self.conv1(x1_1),self.conv1(x2)),dim=1)
        x2_1 = self.conv2(x2_1)
        x2_2 = self.conv1(x1_1)*self.conv1(x2)
        x2_3 = x2_1+x2_2

        x4_2 = self.conv1(x2_3)
        y = self.conv5(x4_2)

        return y,x4_2

class fuse1(nn.Module):
    def __init__(self, channel):
        super(fuse1, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv5 = nn.Conv2d(channel, 1, 1)


    def forward(self, x1, x2):
        x1_1 = x1
        x2_1 = torch.cat((self.conv1(x1_1), self.conv1(x2)), dim=1)
        x2_1 = self.conv2(x2_1)
        x2_2 = self.conv1(x1_1) * self.conv1(x2)
        x2_3 = x2_1 + x2_2

        x4_2 = self.conv1(x2_3)
        y = self.conv5(x4_2)

        return y

class FLCNet(nn.Module):
    def __init__(self):
        super(FLCNet,self).__init__()

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.convbg0 = RFB_modified1(64, 64)
        self.convbg1 = RFB_modified1(256, 64)

        self.convbg4 = RFB_modified(2048, 64)
        self.convbg3 = RFB_modified(1024, 64)
        self.convbg2 = RFB_modified(512, 64)

        self.pd = fuse(64)
        self.pd1 = fuse1(64)

    def forward(self,x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        layer0 = self.resnet.maxpool(x)
        layer1 = self.resnet.layer1(layer0)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)

        x4_1 = self.convbg4(layer4)
        x3_1 = self.convbg3(layer3)
        x2_1 = self.convbg2(layer2)
        x1_1 = self.convbg1(layer1)
        x0_1 = self.convbg0(layer0)

        d4, x4_2 = self.pd(x4_1, x3_1)
        d3, x3_2 = self.pd(x4_2, x2_1)
        d2, x2_2 = self.pd(x3_2, x1_1)
        y = self.pd1(x2_2,x0_1)

        d1 = F.interpolate(y, scale_factor=4, mode='bilinear')
        d2 = F.interpolate(d2, scale_factor=4, mode='bilinear')
        d3 = F.interpolate(d3, scale_factor=8, mode='bilinear')
        d4 = F.interpolate(d4, scale_factor=16, mode='bilinear')

        return  F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3) , F.sigmoid(d4)
