#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import utils

import subprocess
import sys


def to_tensor_dev(val):
    valI=val.astype("float32")
    ret=torch.from_numpy(valI).to(utils.get_device(0))
    ret=ret.to(dtype=torch.float32)
    #ret=torch.tensor(ret,dtype=torch.float32)
    return ret

def marge_onset(pitch_pt, onset_pt, velocity_pt):
    midi=pitch_pt
    ost_est = onset_pt >0.5
    #midi=np.load(midi_path)#[:,3000:4500]#[:,250:500]
    midi_mask=(midi>0.5)

    midi_mask_t=to_tensor_dev(midi_mask)

    midi_onset=(detect_onset(to_tensor_dev(midi_mask)))>0
    midi_onset=midi_onset.to(dtype=torch.float32)

    ost_est_fst=detect_onset(to_tensor_dev(ost_est),wd=-11)>0
    ost_est_fst=ost_est_fst.to(dtype=torch.float32)
    onset_sub=(ost_est_fst-midi_onset)#*midi_mask_t

    ost_est_rep=((detect_onset(onset_sub,wd=11))*midi_mask_t>0.2)
    ost_est_rep=ost_est_rep.to(dtype=torch.float32)*ost_est_fst

    midi_onset=midi_onset+ost_est_rep

    midi_onset3=(detect_onset(midi_onset,wd=7))/3
    midi_onset3=midi_onset3.to(dtype=torch.float32)
    midi_onset=midi_onset.cpu().numpy()

    return midi_mask, midi_onset, velocity_pt

def detect_onset(midi_mask,wd=3):

    w=midi_mask.new_zeros((1,1,1,abs(wd)))+1
    if wd==3:
        w[0,0,0,0]=-1
        w[0,0,0,2]=0
    if wd==5:
        w[0,0,0,3:]=0
    if wd==-5:
        w[0,0,0,:3]=0
    if wd==201:
        w[0,0,0,101:]=0
        for i in range(96):
            w[0,0,0,i]=pow((6/10),(96-i))
    if wd==-11:
        w[0,0,0,:5]=-1
        w[0,0,0,6:]=0
    if wd==11:
        w[0,0,0,:]=1

    pd=int((abs(wd)-1)/2)
    onsetMat=torch.nn.functional.conv1d(midi_mask.view(1,1,88,-1),w,padding=pd)
    onsetMat=onsetMat[:,:,pd:88+pd,:].view(88,-1)
    return onsetMat


def make_IPR(mid, onset, mid_vel,
             out_dir, name):
    ipr_path = out_dir + name + "_ipr.txt"
    hop = 320
    sr = 16000
    pikopiko=False
    beat_time=[]
    one_frame_time=hop/sr
    onMap=onset#np.zeros(np.shape(mid))

    for i in range(np.shape(mid)[0]):
        onFlag=0
        lenCount=0
        onPos=0
        for j in range(np.shape(mid)[1]-1):
            if (mid[i,j]==0 and mid[i,j+1]==1):
                onPos    = j+1
                lenCount += 1
            if mid[i,j]==1 and mid[i,j+1]==1:
                if onMap[i,j+1]>0.1:
                    onMap[i,onPos]=lenCount
                    lenCount = 0
                    onPos = j+1
                    lenCount += 1
                else:
                    lenCount += 1
            if mid[i,j]==1 and mid[i,j+1]==0:
                if j<-1:
                    lenCount=0
                onMap[i,onPos]=lenCount
                lenCount=0
    f = open(ipr_path, 'w')
    c=0
    ofs_p=0
    for i in range(np.shape(mid)[0]):
        ofs_p=0
        for j in range(np.shape(mid)[1]-1):
            if onMap[i,j]!=0:
                vel=int(mid_vel[i,j]*127)
                ons_t=j*one_frame_time#*1200/1202.75536)
                ofs_s=(j+onMap[i,j])*one_frame_time
                if ons_t<ofs_p:
                    ons_t=ofs_p+0.001#float(ons_t)+0.01
                ofs_p=(j+onMap[i,j]-1)*one_frame_time
                if (float(ofs_s)-float(ons_t))<0.03:
                    pass
                else:
                    if vel<40:
                        continue
                    line=make_oneLine(c,
                                      '{:.8f}'.format(ons_t),
                                      '{:.8f}'.format(ofs_s),
                                      i+21,
                                      velOn=vel)
                    c+=1
                    f.write(line)
    if pikopiko:
        subprocess.call(['../PikoPikoRemover/PikoPikoRemover',
                         ipr_path,
                         ipr_path[:-4]+"_pr.txt"])
        ipr_path=ipr_path[:-4]+"_pr.txt"
    return ipr_path

def make_oneLine(line_no,
                 on_time,
                 off_time,
                 nort_id,
                 velOn,
                 part_id=0):
    velOff=velOn
    write_line1=str(line_no) \
                 +"\t"+str(on_time)\
                 +"\t"+str(off_time)\
                 +"\t"+str(nort_id)\
                 +"\t"+str(velOn)\
                 +"\t"+str(velOff)\
                 +"\t"+str(part_id)+"\n"
    return write_line1

def main(pitch_pt, onset_pt, velocity_pt,
         input_path, out_dir, name):
    #name = input_path.split(".")[-2]
    #name = ".".join(input_path.split(".")[:-1])
    midi_path = out_dir + name + ".mid"
    midi_mask, midi_onset, vel_mid = marge_onset(pitch_pt,
                                                 onset_pt,
                                                 velocity_pt)
    ipr_path = make_IPR(midi_mask, midi_onset, vel_mid,
                        out_dir, name)
    returncode = subprocess.call(
        ['./pianoroll2midi',
         ipr_path,
         midi_path,
         "track0:1"])
    return midi_path


