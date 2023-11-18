#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy
from scipy import signal

import utils


class stft2hcfp(nn.Module):
    def __init__(self):
        super(stft2hcfp, self).__init__()
        self.CFP_filterbank =\
            CFP_filterbank_C(
                fr=2.0,
                fs=16000,
                Hop=320,
                h= scipy.signal.blackmanharris(2049),
                fc=27.5,
                tc=1/4487.0,
                g=np.array([0.24, 0.6, 1]),
                NumPerOctave=48)
        self.num_harmonic=5
        self.start_freq=27.5
        self.num_per_octave=48
        self.bins_per_note = int(self.num_per_octave / 12)
        self.total_bins = int(self.bins_per_note * 88)


    def fetch_harmonic(self,data, cenf, ith_har,
                       is_reverse=False):
        ith_har += 1
        if ith_har != 0 and is_reverse:
            ith_har = 1/ith_har

        hid = min(range(len(cenf)), key=lambda i: abs(cenf[i]-ith_har*self.start_freq))
        hid_r=min(range(len(cenf)), key=lambda i: abs(cenf[i]-1/ith_har*self.start_freq))
        #harmonic = torch.zeros((data.shape[0],self.total_bins, data.shape[2])).to(utils.get_device(0))
        harmonic = data.new_zeros((data.shape[0],self.total_bins, data.shape[2]))
        if is_reverse:
            upper_bound = min(len(cenf)-1, self.total_bins-hid_r)
            harmonic[:,hid_r:] = data[:,hid:upper_bound]
        else:
            upper_bound = min(len(cenf)-1, hid+self.total_bins)
            harmonic[:,:(upper_bound-hid)] = data[:,hid:upper_bound]
        return harmonic

    def feature_extraction(self, dx):
        tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = self.CFP_filterbank(dx)
        Z = tfrLF * tfrLQ
        return Z, tfrL0, tfrLF, tfrLQ, t, CenFreq, f

    def forward(self,dx):

        dx=torch.transpose(dx,2,1).to(utils.get_device(0))
        #print(dx)
        out=self.feature_extraction(dx)
        cenf = out[5]
        har = []
        for i in range(self.num_harmonic+1):
            har.append(self.fetch_harmonic(out[1], cenf, i))
        har_s = torch.stack(har,dim=1)
        har = []

        for i in range(self.num_harmonic+1):
            har.append(self.fetch_harmonic(out[3], cenf, i, is_reverse=True))

        har_c = torch.stack(har,dim=1)
        piece = torch.cat([har_s, har_c],dim=1)

        piece=torch.transpose(piece,3,2)
        return piece

class Freq2LogFreqMapping_C(nn.Module):
    def __init__(self,f, fr, fc, tc, NumPerOct):
        super(Freq2LogFreqMapping_C, self).__init__()
        self.f =f
        self.fr=fr
        self.fc=fc
        self.tc=tc
        self.NumPerOct=NumPerOct
        self.StartFreq = self.fc
        self.StopFreq = 1/self.tc
        self.Nest = int(np.ceil(np.log2(self.StopFreq/self.StartFreq))*self.NumPerOct)
        self.central_freq = []

        for i in range(0, self.Nest):
            self.CenFreq = self.StartFreq*pow(2, float(i)/self.NumPerOct)
            if self.CenFreq < self.StopFreq:
                self.central_freq.append(self.CenFreq)
            else:
                break

        self.Nest = len(self.central_freq)
        self.freq_band_transformation = torch.zeros((self.Nest-1, len(self.f)), dtype=torch.float).to(utils.get_device(0))
        for i in range(1, self.Nest-1):
            self.l = int(round(self.central_freq[i-1]/self.fr))
            self.r = int(round(self.central_freq[i+1]/self.fr)+1)
            #rounding1
            if self.l >= self.r-1:
                self.freq_band_transformation[i, self.l] = 1
            else:
                for j in range(self.l, self.r):
                    if self.f[j] > self.central_freq[i-1] and self.f[j] < self.central_freq[i]:
                        self.freq_band_transformation[i, j] = (self.f[j] - self.central_freq[i-1]) / (self.central_freq[i] - self.central_freq[i-1])
                    elif self.f[j] > self.central_freq[i] and self.f[j] < self.central_freq[i+1]:
                        self.freq_band_transformation[i, j] = (self.central_freq[i + 1] - self.f[j]) / (self.central_freq[i + 1] - self.central_freq[i])

    def forward(self, tfr):
        self.freq_band_transformation=self.freq_band_transformation.to(utils.get_device(0))
        tfr=tfr.to(utils.get_device(0))
        #print(self.freq_band_transformation,"a")
        #print(tfr,"b")
        self.tfrL = torch.matmul(self.freq_band_transformation[None,:,:], tfr)
        return self.tfrL, self.central_freq


class Quef2LogFreqMapping_C(nn.Module):
    def __init__(self, q, fs, fc, tc, NumPerOct):
        super(Quef2LogFreqMapping_C, self).__init__()
        self.q   =q
        self.fs  =fs
        self.fc  =fc
        self.tc  =tc
        self.NumPerOct =NumPerOct
        self.StartFreq = self.fc
        self.StopFreq = 1/self.tc
        self.Nest = int(np.ceil(np.log2(self.StopFreq/self.StartFreq))*self.NumPerOct)
        self.central_freq = []
        for i in range(0, self.Nest):
            self.CenFreq = self.StartFreq*pow(2, float(i)/self.NumPerOct)
            if self.CenFreq < self.StopFreq:
                self.central_freq.append(self.CenFreq)
            else:
                break
        self.f = 1/self.q
        self.Nest = len(self.central_freq)
        self.freq_band_transformation = torch.zeros((self.Nest-1, len(self.f)), dtype=torch.float32).to(utils.get_device(0))
        for i in range(1, self.Nest-1):
            for j in range(int(round(self.fs/self.central_freq[i+1])), int(round(self.fs/self.central_freq[i-1])+1)):
                if self.f[j] > self.central_freq[i-1] and self.f[j] < self.central_freq[i]:
                    self.freq_band_transformation[i, j] = (self.f[j] - self.central_freq[i-1])/(self.central_freq[i] - self.central_freq[i-1])
                elif self.f[j] > self.central_freq[i] and self.f[j] < self.central_freq[i+1]:
                    self.freq_band_transformation[i, j] = (self.central_freq[i + 1] - self.f[j]) / (self.central_freq[i + 1] - self.central_freq[i])

    def forward(self, ceps):
        self.freq_band_transformation=self.freq_band_transformation.to(utils.get_device(0))
        ceps=ceps.to(utils.get_device(0))
        self.tfrL = torch.matmul(self.freq_band_transformation[None,:, :ceps.size()[1]], ceps)
        #self.tfrL = np.dot(self.freq_band_transformation[:, :len(ceps)], ceps)
        #my_plot.my_imshow(ceps[0].detach().cpu().numpy())
        #plt.savefig("o5")
        return self.tfrL, self.central_freq


class CFP_filterbank_C(nn.Module):
    def __init__(self,fr, fs, Hop, h, fc, tc, g, NumPerOctave):
        super(CFP_filterbank_C, self).__init__()
        self.fr  =fr
        self.fs  =fs
        self.Hop =Hop
        self.h   =h
        self.fc  =fc
        self.tc  =tc
        self.g   =g
        self.NumPerOctave=NumPerOctave
        self.NumofLayer = np.size(self.g)
        #[tfr, f, t, N]
        self.lenX=Hop*512

        self.t = np.arange(Hop, np.ceil(self.lenX/float(Hop))*Hop, Hop)
        self.N = int(fs/float(fr))
        self.sqrtN = np.sqrt(self.N)
        self.window_size = len(h)
        self.f = fs*np.linspace(0, 0.5, int(np.round(self.N/2)),
                                endpoint=True)
        self.halfN=int(round(self.N/2))
        self.rFs_p_Tc=round(self.fs*self.tc)
        self.rFc_d_Tr=round(self.fc/self.fr)
        self.HighFreqIdx = int(round((1/self.tc)/self.fr)+1)
        self.HighQuefIdx = int(round(self.fs/self.fc))+1
        self.q = np.arange(self.HighQuefIdx)/float(self.fs)+0.000001

        self.Freq2LogFreqMapping=Freq2LogFreqMapping_C(self.f[:self.HighFreqIdx], self.fr, self.fc, self.tc, self.NumPerOctave)
        self.Quef2LogFreqMapping=Quef2LogFreqMapping_C(self.q, self.fs, self.fc, self.tc, self.NumPerOctave)

    def nonlinear_func(self,X, g, cutoff):
        cutoff = int(cutoff)
        if g!=0:
            X[X<0] = 0
            X[:,:cutoff, :] = 0
            X[:,-cutoff:, :] = 0
            X = torch.pow(X, g)
        else:
            X = torch.log(X)
            X[:,:cutoff, :] = 0
            X[:,-cutoff:, :] = 0
        return X

    def forward(self,tfr):
        #print("st_make_hcfp_fw")
        tfr0 = tfr.to(utils.get_device(0)) # original STFT
        ceps = tfr0.new_zeros(tfr.shape)
        #ceps = torch.zeros(tfr.shape)
        tfr_a = tfr0.new_zeros((ceps.shape[0],ceps.shape[1],ceps.shape[2],2))
        ceps_b= tfr0.new_zeros((ceps.shape[0],ceps.shape[1],ceps.shape[2],2))
        if self.NumofLayer >= 2:
            for gc in range(1, self.NumofLayer):
                if np.remainder(gc, 2) == 1:
                    tc_idx = self.rFs_p_Tc
                    tfr_a[:,:,:,0]=tfr
                    tfr_a=torch.transpose(tfr_a,2,1)
                    ceps = torch.fft(tfr_a, 1)[:,:,:,0]/self.sqrtN
                    ceps = torch.transpose(ceps,2,1)
                    ceps = self.nonlinear_func(ceps, self.g[gc], tc_idx)
                else:
                    fc_idx = self.rFc_d_Tr
                    ceps_b[:,:,:,0]=ceps
                    ceps_b=torch.transpose(ceps_b,2,1)
                    #ceps=ceps_b
                    #tfr = torch.ifft(ceps_b, 2)
                    tfr = torch.fft(ceps_b, 1)[:,:,:,0]/self.sqrtN
                    tfr = torch.transpose(tfr,2,1)
                    tfr = self.nonlinear_func(tfr, self.g[gc], fc_idx)
        #print("md_make_hcfp_fw")
        tfr0 = tfr0[:,:self.halfN,:]
        tfr = tfr[:,:self.halfN,:]
        ceps = ceps[:,:self.halfN,:]

        f = self.f[:self.HighFreqIdx]
        tfr0 = tfr0[:,:self.HighFreqIdx,:]
        tfr = tfr[:,:self.HighFreqIdx,:]

        q=self.q
        fr=self.fr
        fc=self.fc
        tc=self.tc
        fs=self.fs
        NumPerOctave=self.NumPerOctave
        ceps = ceps[:,:self.HighQuefIdx,:]
        tfrL0, central_frequencies = self.Freq2LogFreqMapping(tfr0)
        tfrLF, central_frequencies = self.Freq2LogFreqMapping(tfr)
        tfrLQ, central_frequencies = self.Quef2LogFreqMapping(ceps)
        #print("ed_make_hcfp_fw")
        return tfrL0, tfrLF, tfrLQ, f, q, self.t, central_frequencies
