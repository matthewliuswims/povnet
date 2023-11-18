#!/usr/bin/env python
# -*- coding: utf-8 -*
import subprocess
import sys
import os
import soundfile as sf
import numpy as np
import povnet
import pov2pMIDI
import scipy
import torch
from scipy import signal



class Wav2spec():
    def __init__(self, wav_path):
        self.wav_path = wav_path
        self.wav = None
        self.spec = None
        self.fs_should_be = 16000.0
        self.segmented_spec=[]

    def __call__(self):
        self.wav_read()
        self.compute_stft()
        self.cutUnitSeg_tensor()
        return self.segmented_spec[:,0,:,:]

    def set_wav(self,wav):
        self.wav=wav


    def get_wav_from_youtube(self,url):
        exit("TBI")

    def wav_read(self):
        x, fs = sf.read(self.wav_path)
        #if stereo, adopt mean
        if len(x.shape)>1:
            x = np.mean(x, axis = 1)
        if fs != self.fs_should_be:
            x = signal.resample_poly(x, self.fs_should_be, fs)
        x = x.astype('float32')
        self.wav = x

    def STFT(self, x, fr, fs, Hop, h):
        t = np.arange(Hop, np.ceil(len(x)/float(Hop))*Hop, Hop)
        N = int(fs/float(fr))
        window_size = len(h)
        f = fs*np.linspace(0, 0.5,
                           int(np.round(N/2)),
                           endpoint=True)
        Lh = int(np.floor(float(window_size-1) / 2))
        tfr = np.zeros((int(N), len(t)), dtype=np.float)

        for icol in range(0, len(t)):
            ti = int(t[icol])
            tau = np.arange(int(-min([round(N/2.0)-1, Lh, ti-1])), \
                            int(min([round(N/2.0)-1, Lh, len(x)-ti])))
            indices = np.mod(N + tau, N) + 1
            tfr[indices-1, icol] = x[ti+tau-1] * h[Lh+tau-1] \
                                    /np.linalg.norm(h[Lh+tau-1])

        #tfr = abs(scipy.fftpack.fft(tfr, n=N, axis=0))
        tfr = abs(scipy.fft.fft(tfr, n=N, axis=0))
        return tfr, f, t, N

    def compute_stft(self):
        Hop=320
        w=2049
        fr=2.0
        fc=27.5
        tc=1/4487.0
        g=[0.24, 0.6, 1]
        NumPerOctave=48

        h = scipy.signal.blackmanharris(w) # window size
        g = np.array(g)
        NumofLayer = np.size(g)

        [tfr, f, t, N] = self.STFT(self.wav, fr, self.fs_should_be, Hop, h)
        tfr = np.power(abs(tfr), g[0])

        spec=tfr[:int(np.shape(tfr)[0]/2),:]
        specT=np.zeros((np.shape(spec)[0],np.shape(spec)[1]+128))
        specT[:,128:]=spec
        spec = specT[None,:,:]
        self.spec = spec

    def cutUnitSeg_tensor(self,ovl=128,seg_size=512):
        tempSeg=np.zeros(
            [np.shape(self.spec)[0],
             np.shape(self.spec)[1],
             (np.shape(self.spec)[2]//(seg_size)+2)*seg_size])
        tempSeg[:,:,:np.shape(self.spec)[2]]=abs(self.spec)
        if ovl==0:
            segNum=np.shape(tempSeg)[2]//seg_size
        else:
            segNum=(np.shape(tempSeg)[2]//(seg_size-ovl))-1
        self.segmented_spec=np.stack(
            [tempSeg[:,:,i*(seg_size-ovl):i*(seg_size-ovl)+seg_size]
             for i in range(segNum)])


def main(wav_path, out_dir, name, gpuN=0):
    wav2spec = Wav2spec(wav_path)
    segmented_spec = wav2spec()
    pitch_pt, onset_pt, velocity_pt = \
        povnet.povnet_main(segmented_spec, gpuN)
    midi_path = pov2pMIDI.main(pitch_pt,
                               onset_pt,
                               velocity_pt,
                               wav_path,
                               out_dir,
                               name)
    return midi_path



if __name__=="__main__":
    args=sys.argv
    if len(args) <= 1:
        print("python wav2score.py",
              "input_file.wav",
              "output_dir",
              "output_filename",
              "gpu_num(optional)")
        exit()

    #wav_path = os.path.abspath(args[1])
    wav_path = args[1]
    if len(args) >= 3:
        out_dir = os.path.abspath(args[2])+"/"
        os.makedirs(out_dir, exist_ok=True)
        print("outdir = ", out_dir)
    else:
        out_dir = os.path.abspath("./output/")
        print("outdir = ./output/(default)")
    if len(args) >= 4:
        name = args[3]
    else:
        if wav_path.split(".")[-1]!="wav":
            name = "temp"
        else:
            #name = wav_path.split("/")[-1].split(".")[0]
            name = os.path.abspath(wav_path).split("/")[-1].split(".")[0]
    out_dir = out_dir + name + "/"
    #cmd = ["rm", "-r", out_dir]
    #subprocess.run(cmd)
    os.makedirs(out_dir,exist_ok=True)

    if wav_path.split(".")[-1]!="wav":
        print("This script accept only wav files.")
        exit()
    wav_path = os.path.abspath(wav_path)
    if len(args) >= 5:
        gpuN = int(args[4])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuN)
    cuda = torch.cuda.is_available()
    if cuda:
        print("GPU")
    else:
        print("CPU")
    #print(wav_path, out_dir, name)
    main(wav_path, out_dir, name, gpuN)
