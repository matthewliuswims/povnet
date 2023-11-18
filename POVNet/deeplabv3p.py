#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import utils


class conv_blockT(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,strides=(2, 2),dilation_rate=1,dropout_rate=0.2):
        super().__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.kernel_size=kernel_size
        self.strides=strides
        self.dilation_rate=dilation_rate
        self.dropout_rate=dropout_rate
        self.input_tensor_size=(512,352)
        self.norm1=nn.BatchNorm2d(num_features=self.in_channel)
        self.norm2=nn.BatchNorm2d(num_features=self.out_channel)
        self.dout=nn.Dropout(self.dropout_rate)
        self.cnn1=nn.Conv2d(in_channels=self.in_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size,
                            stride=self.strides,
                            padding=tuple([np.ceil((self.strides[0]*(self.input_tensor_size[0]-1)-self.input_tensor_size[0]+self.dilation_rate*(self.kernel_size[0]-1)+1)/2).astype(int),
                                           np.ceil((self.strides[1]*(self.input_tensor_size[1]-1)-self.input_tensor_size[1]+self.dilation_rate*(self.kernel_size[1]-1)+1)/2).astype(int)]),
                            dilation=self.dilation_rate)
        self.cnn2=nn.Conv2d(in_channels=self.out_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size,
                            stride=(1,1),
                            padding=tuple([np.ceil(((self.input_tensor_size[0]-1)-self.input_tensor_size[0]+self.dilation_rate*(self.kernel_size[0]-1)+1)/2).astype(int),
                                           np.ceil(((self.input_tensor_size[1]-1)-self.input_tensor_size[1]+self.dilation_rate*(self.kernel_size[1]-1)+1)/2).astype(int)]),
                            dilation=self.dilation_rate)
        self.cnn3=nn.Conv2d(in_channels=self.in_channel,
                            out_channels=self.out_channel,
                            kernel_size=(1,1),
                            stride=self.strides,
                            padding=tuple([np.ceil((self.strides[0]*(self.input_tensor_size[0]-1)-self.input_tensor_size[0]+1)/2).astype(int),
                                           np.ceil((self.strides[1]*(self.input_tensor_size[1]-1)-self.input_tensor_size[1]+1)/2).astype(int)]))


    def forward(self, input_tensor):
        skip = input_tensor
        input_tensor = self.dout(self.norm1(F.relu(input_tensor)))
        input_tensor = self.cnn1(input_tensor)
        input_tensor = self.dout(self.norm2(F.relu(input_tensor)))

        input_tensor = self.cnn2(input_tensor)

        if (self.strides != (1, 1)):
            skip = self.cnn3(skip)
        input_tensor = input_tensor+ skip

        return input_tensor

class transpose_conv_blockT(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,strides=(2, 2),dropout_rate=0.4):
        super().__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.kernel_size=kernel_size
        self.input_tensor_size=(512,352)
        self.strides=strides
        self.dropout_rate=dropout_rate
        self.cnn1=nn.ConvTranspose2d(in_channels=self.in_channel,
                                     out_channels=self.out_channel,
                                     kernel_size=self.kernel_size,
                                     stride=(1,1),
                                     padding=tuple([np.ceil(((self.input_tensor_size[0]-1)-self.input_tensor_size[0]+(self.kernel_size[0]-1)+1)/2).astype(int),
                                                    np.ceil(((self.input_tensor_size[1]-1)-self.input_tensor_size[1]+(self.kernel_size[1]-1)+1)/2).astype(int)]))
        self.cnn2=nn.ConvTranspose2d(in_channels=self.out_channel,
                                     out_channels=self.out_channel,
                                     kernel_size=self.kernel_size,
                                     stride=self.strides,
                                     output_padding=(1,1),
                                     padding=tuple([np.ceil((self.strides[0]*(self.input_tensor_size[0]-1)-self.input_tensor_size[0]+(self.kernel_size[0]-1)+1)/2).astype(int),
                                                    np.ceil((self.strides[1]*(self.input_tensor_size[1]-1)-self.input_tensor_size[1]+(self.kernel_size[1]-1)+1)/2).astype(int)])
        )
        self.cnn3=nn.ConvTranspose2d(in_channels=self.in_channel,
                                     out_channels=self.out_channel,
                                     kernel_size=(1,1),
                                     stride=self.strides,
                                     output_padding=(1,1),
                                     padding=tuple([np.ceil((self.strides[0]*(self.input_tensor_size[0]-1)-self.input_tensor_size[0]+1)/2).astype(int),
                                                    np.ceil((self.strides[1]*(self.input_tensor_size[1]-1)-self.input_tensor_size[1]+1)/2).astype(int)]))

        self.norm1=nn.BatchNorm2d(num_features=self.in_channel)
        self.norm2=nn.BatchNorm2d(num_features=self.out_channel)
        self.dout=nn.Dropout(self.dropout_rate)

    def forward(self, input_tensor):
        skip = input_tensor

        input_tensor = self.dout(self.norm1(F.relu(input_tensor)))
        input_tensor = self.cnn1(input_tensor)
        input_tensor = self.dout(self.norm2(F.relu(input_tensor)))

        input_tensor = self.cnn2(input_tensor)

        if (self.strides != (1, 1)):
            skip = self.cnn3(skip)
        input_tensor = input_tensor+skip
        return input_tensor


class segT(nn.Module):
    def __init__(self,
                 feature_num=128,
                 timesteps=256,
                 multi_grid_layer_n=1,
                 multi_grid_n=3,
                 input_channel=2,
                 prog = False,
                 out_class=2):
        self.in_channel =input_channel
        self.out_class=out_class
        super().__init__()
        self.multi_grid_layer_n=multi_grid_layer_n
        self.multi_grid_n=multi_grid_n

        self.cnn0=nn.Conv2d(in_channels=self.in_channel,
                            out_channels=2**5,
                            kernel_size=(7,7),
                            stride=(1,1),
                            padding=(3,3))

        self.conv1_1=conv_blockT(2 ** 5, 2 ** 5, (3, 3), strides=(2, 2))
        self.conv1_2=conv_blockT(2 ** 5, 2 ** 5, (3, 3), strides=(1, 1))

        self.conv2_1=conv_blockT(2 ** 5, 2 ** 6, (3, 3), strides=(2, 2))
        self.conv2_2=conv_blockT(2 ** 6, 2 ** 6, (3, 3), strides=(1, 1))
        self.conv2_3=conv_blockT(2 ** 6, 2 ** 6, (3, 3), strides=(1, 1))

        self.conv3_1=conv_blockT(2 ** 6, 2 ** 7, (3, 3), strides=(2, 2))
        self.conv3_2=conv_blockT(2 ** 7, 2 ** 7, (3, 3), strides=(1, 1))
        self.conv3_3=conv_blockT(2 ** 7, 2 ** 7, (3, 3), strides=(1, 1))
        self.conv3_4=conv_blockT(2 ** 7, 2 ** 7, (3, 3), strides=(1, 1))

        self.conv4_1=conv_blockT(2 ** 7, 2 ** 8, (3, 3), strides=(2, 2))
        self.conv4_2=conv_blockT(2 ** 8, 2 ** 8, (3, 3), strides=(1, 1))
        self.conv4_3=conv_blockT(2 ** 8, 2 ** 8, (3, 3), strides=(1, 1))
        self.conv4_4=conv_blockT(2 ** 8, 2 ** 8, (3, 3), strides=(1, 1))
        self.conv4_5=conv_blockT(2 ** 8, 2 ** 8, (3, 3), strides=(1, 1))

        self.norm1=nn.BatchNorm2d(num_features=self.in_channel)

        self.norm_a1=nn.BatchNorm2d(num_features=2 ** 8)
        self.dout_a1=nn.Dropout(0.3)
        self.conv_a1=nn.Conv2d(in_channels=2 ** 8,
                               out_channels=2 ** 9,
                               kernel_size=(1, 1),
                               stride=(1,1),
                               padding=(0,0))
        self.norm_a2=nn.BatchNorm2d(num_features=2 ** 9)

        self.norm_b1=nn.BatchNorm2d(num_features=2 ** 9)
        self.dout_c1=nn.Dropout(0.3)
        self.conv_c1=nn.Conv2d(in_channels=2 ** 9*4,
                               out_channels=2 ** 9,
                               kernel_size=(1, 1),
                               stride=(1,1),
                               padding=(0,0))

        self.norm_d0=nn.BatchNorm2d(num_features=2 ** 9)
        self.conv_d0=nn.Conv2d(in_channels=2 ** 9,
                               out_channels=2 ** 8,
                               kernel_size=(1, 1),
                               stride=(1,1),
                               padding=(0,0))

        self.convT1_0=transpose_conv_blockT(2 ** 8,2 ** 7, (3, 3), strides=(2, 2))
        self.norm_d1=nn.BatchNorm2d(num_features=2 ** 7)
        self.norm_d2=nn.BatchNorm2d(num_features=2 ** 7)
        self.dout_d2=nn.Dropout(0.4)
        self.conv_d2=nn.Conv2d(in_channels=2 ** 7*2,
                               out_channels=2 ** 7,
                               kernel_size=(1, 1),
                               stride=(1,1),
                               padding=(0,0))
        self.convT1_1=transpose_conv_blockT(2 ** 7,2 ** 6, (3, 3), strides=(2, 2))

        self.norm_e1=nn.BatchNorm2d(num_features=2 ** 6)
        self.norm_e2=nn.BatchNorm2d(num_features=2 ** 6)
        self.dout_e2=nn.Dropout(0.4)
        self.conv_e2=nn.Conv2d(in_channels=2 ** 6*2,
                               out_channels=2 ** 6,
                               kernel_size=(1, 1),
                               stride=(1,1),
                               padding=(0,0))
        self.convT2_1=transpose_conv_blockT(2 ** 6,2 ** 5, (3, 3), strides=(2, 2))

        self.norm_f1=nn.BatchNorm2d(num_features=2 ** 5)
        self.norm_f2=nn.BatchNorm2d(num_features=2 ** 5)
        self.dout_f2=nn.Dropout(0.4)
        self.conv_f2=nn.Conv2d(in_channels=2 ** 5*2,
                               out_channels=2 ** 5,
                               kernel_size=(1, 1),
                               stride=(1,1),
                               padding=(0,0))
        self.convT3_1=transpose_conv_blockT(2 ** 5,2 ** 5, (3, 3), strides=(2, 2))

        self.norm_g1=nn.BatchNorm2d(num_features=2 ** 5)
        self.dout_g1=nn.Dropout(0.4)
        self.conv_g1=nn.Conv2d(in_channels=2 ** 5,
                               out_channels=self.out_class,
                               kernel_size=(1, 1),
                               stride=(1,1),
                               padding=(0,0))

    def forward(self, input_tensor):
        layer_out = []

        en = self.cnn0(input_tensor)
        layer_out.append(en)

        en_l1 = self.conv1_1(en)

        en_l1 = self.conv1_2(en_l1)
        layer_out.append(en_l1)

        en_l2 = self.conv2_1(en_l1)
        en_l2 = self.conv2_2(en_l2)
        en_l2 = self.conv2_3(en_l2)
        layer_out.append(en_l2)

        en_l3 = self.conv3_1(en_l2)
        en_l3 = self.conv3_2(en_l3)
        en_l3 = self.conv3_3(en_l3)
        en_l3 = self.conv3_4(en_l3)
        layer_out.append(en_l3)

        en_l4 = self.conv4_1(en_l3)
        en_l4 = self.conv4_2(en_l4)
        en_l4 = self.conv4_3(en_l4)
        en_l4 = self.conv4_4(en_l4)
        en_l4 = self.conv4_5(en_l4)
        layer_out.append(en_l4)

        feature = en_l4

        for i in range(self.multi_grid_layer_n):
            feature = self.dout_a1(self.norm_a1(F.relu(feature)))
            m = self.norm_a2(F.relu(self.conv_a1(feature)))
            multi_grid = m
            for ii in range(self.multi_grid_n):
                conv_b1=nn.Conv2d(in_channels=2**8,
                                  out_channels=2**9,
                                  kernel_size=(3,3),
                                  stride=(1,1),
                                  padding=tuple([int(+2**ii),
                                                 int(+2**ii)]),
                                  dilation=2**ii).to(utils.get_device(0))
                m=F.relu(conv_b1(feature))
                m = self.norm_b1(m)
                multi_grid = torch.cat((multi_grid, m),dim=1)##dim?? axis???
            multi_grid = self.dout_c1(multi_grid)
            feature = self.conv_c1(multi_grid)
            layer_out.append(feature)

        feature = self.norm_d0(F.relu(feature))
        feature = self.conv_d0(feature)
        #feature = add([feature, en_l4])
        feature = feature+en_l4
        de_l1 = self.convT1_0(feature)
        layer_out.append(de_l1)
    
        skip = de_l1
        de_l1 = self.norm_d1(F.relu(de_l1))
        de_l1 = torch.cat((de_l1,self.norm_d2(F.relu(en_l3))),dim=1)#dim??
        de_l1 = self.dout_d2(de_l1)
        de_l1 = self.conv_d2(de_l1)
        de_l1 = de_l1+skip#add([de_l1, skip])
        de_l2 = self.convT1_1(de_l1)
        layer_out.append(de_l2)
    
        skip = de_l2
        de_l2 = self.norm_e1(F.relu(de_l2))
        de_l2 = torch.cat((de_l2,self.norm_e2(F.relu(en_l2))),dim=1)#dim??
        de_l2 = self.dout_e2(de_l2)
        de_l2 = self.conv_e2(de_l2)
        de_l2 = de_l2+skip#add([de_l1, skip])
        de_l3 = self.convT2_1(de_l2)
        layer_out.append(de_l3)
    
        skip = de_l3
        de_l3 = self.norm_f1(F.relu(de_l3))
        de_l3 = torch.cat([de_l3,self.norm_f2(F.relu(en_l1))],dim=1)#dim??
        de_l3 = self.dout_f2(de_l3)
        de_l3 = self.conv_f2(de_l3)
        de_l3 = de_l3+skip#add([de_l1, skip])

        de_l4 = self.convT3_1(de_l3)
        layer_out.append(de_l4)

    
        de_l4 = self.norm_g1(F.relu(de_l4))
        de_l4 = self.dout_g1(de_l4)
        out = self.conv_g1(de_l4)
        #out = Conv2D(out_class, (1, 1), strides=(1, 1), padding="same", name='prediction')(de_l4)
        return out

