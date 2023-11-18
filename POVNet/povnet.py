#!/usr/bin/env python
# -*- coding: utf-8 -*
import os
import numpy as np
import math
import torch
import torch.nn as nn
import deeplabv3p
import cfpLayer
import utils
from torchvision import transforms


class freq2midMAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.dev=utils.get_device(0)
        self.wMat=torch.zeros(88,352).to(self.dev)
        for i in range(88):
            self.wMat[i,i*4+1]=1

    def forward(self, ts):
        ret=torch.matmul(self.wMat[None,:,:],torch.transpose(ts,2,1))
        ret=torch.transpose(ret,2,1)
        return ret

class PitchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.segT = deeplabv3p.segT(timesteps=512,
                                    input_channel=12,
                                    out_class=1)
        self.make_hcfp = cfpLayer.stft2hcfp()
        self.line = nn.Linear(352, 88)
        self.cnnLAST = nn.Conv2d(1, 1, (1,4),
                                 padding=(0,0),
                                 stride=(1,4))
        self.freq2midMATF = freq2midMAT()

    def forward(self, xs):
        btcS = xs.shape[0]

        h0 = self.make_hcfp(xs)

        y = self.segT(h0)

        y = y.view(btcS, -1, 352)
        return y, y

class OnVelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.segT = deeplabv3p.segT(timesteps=512,
                               input_channel=12,
                               out_class=2)
        self.make_hcfp = cfpLayer.stft2hcfp()
        self.line = nn.Linear(352, 88)
        self.cnnLAST = nn.Conv2d(1, 1, (1,4),
                                 padding = (0, 0),
                                 stride = (1,4))
        self.freq2midMATF = freq2midMAT()

    def forward(self, xs):
        btcS = xs.shape[0]
        h0 = self.make_hcfp(xs)
        y = self.segT(h0)
        os = y[:, 0].view(btcS, -1, 352)
        vel = y[:,1].view(btcS, -1, 352)
        return os, vel

def povnet_main(segmented_spec, gpuN=0):


    def data_pre_process(segmented_spec,
                         batch_size=2):
        seg_num = np.shape(segmented_spec)[0]
        batch_num = int(math.ceil(seg_num / batch_size))

        f_len = np.shape(segmented_spec)[1]*2
        t_len = np.shape(segmented_spec)[2]
        segmented_spec = segmented_spec.astype("float32")
        spec = np.zeros((np.shape(segmented_spec)[0],
                       np.shape(segmented_spec)[1]*2,
                       np.shape(segmented_spec)[2]),
                      dtype=np.float32)
        freq_half = int(np.shape(spec)[1]/2)
        spec[:,:freq_half,:] = segmented_spec
        spec[:,freq_half:,:] = segmented_spec

        segmented_spec = torch.from_numpy(
            np.transpose(spec,
                         (0, 2, 1)))

        segmented_spec_batch =[]
        for i in range(batch_num):
            segmented_spec_batch.append(segmented_spec[i*batch_size:(i+1)*batch_size,:,:])
        return segmented_spec_batch

    def initialize(
            device,
            mode="pitch"):
        if mode=="onvel":
            model = OnVelNet().to(device)
            model = torch.nn.DataParallel(model)
            model_param_path="../from_sd2/output/20190929_163529/ep71net.bin"
            #model_param_path = "./params/OnVelNet/ep71net.bin"
            model.load_state_dict(torch.load(model_param_path))

        else:
            model = PitchNet().to(device)
            model = torch.nn.DataParallel(model)
            model_param_path="../from_sd2//output/20190921_180708/ep197net.bin"
            #model_param_path = "./params/pitchNet/ep197net.bin"
            model.load_state_dict(torch.load(
                model_param_path,
                map_location=utils.get_device(0)))
                #map_location=torch.device('cpu')))
        return model

    def povnet_proc(data_loader, model, mode="pitch"):
        model.eval()
        st=64
        ed=512-st
        resStack=[]
        resStackSpec=[]
        resStackVEL=[]
        for xs in data_loader:
            with torch.no_grad():
                mid,os=model.forward(xs)
                mid=torch.sigmoid(mid)#[:,0,:,:]
                os=torch.clamp(os,0,1)
                if mid.shape[2]==352:
                    mid=model.module.freq2midMATF(mid)
                    os=model.module.freq2midMATF(os)
                midE_M = np.transpose(mid.detach().cpu().numpy(),
                                      (0,2,1))[:,:,st:ed]#[20:108,:]
                osEST = np.transpose(os.detach().cpu().numpy(),
                                     (0,2,1))[:,:,st:ed]#[20:108,:]
                for midi,vel_est in zip(midE_M,osEST):
                    resStack.append(midi)
                    resStackVEL.append(vel_est)

        resENT = np.zeros((88,
                           np.shape(resStack)[0]*(512-128)))
        resENTvel = np.zeros((88,
                              np.shape(resStackVEL)[0]*(512-128)),
                             dtype=float)
        for i,(midi,vel_est) in enumerate(zip(resStack,resStackVEL)):
            resENTvel[:,i*(512-128):(i+1)*(512-128)]=vel_est
            resENT[:,i*(512-128):(i+1)*(512-128)]=midi

        if mode=="pitch":
            resENT=resENT[:,64:]
            return resENT

        elif mode=="onvel":
            resENT=resENT[:,64:]
            resENTvel=resENTvel[:,64:]
            return resENT, resENTvel
        else:
            exit("error")

    device = utils.get_device(0)
    #config = toml.load(args.config)
    model = initialize(device)
    segmented_spec_tensor = data_pre_process(segmented_spec)
    pitch_pt = povnet_proc(segmented_spec_tensor,
                           model,
                           mode="pitch")

    del model
    del segmented_spec_tensor
    model = initialize(device,mode="onvel")
    segmented_spec_tensor = data_pre_process(segmented_spec)

    onset_pt, velocity_pt = povnet_proc(segmented_spec_tensor,
                                        model,mode="onvel")
    #model.cpu()
    del model
    del segmented_spec_tensor
    return pitch_pt, onset_pt, velocity_pt
