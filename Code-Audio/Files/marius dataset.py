#code for insectsoundMarius dataset, as prep remove the .DS_store files which lack function and are annoying to filter out

import pandas as pd
import csv
import os #for file navigation

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_PATH)             #ensure all following commands are running relative to the location of this file

#fairly simple way to get all filepaths of files in the folder of the code file and all subfolders
#then easily filter-able given the consistent syntax to get specific files
def getfilepaths(inputdir):
    filelist = list()
    def listfiles(rootdir): #recursive function to get filenames of everything in the root path and every subfolder
        for fileorfoldername in os.scandir(rootdir):
            if fileorfoldername.is_dir():
                filelist.append(fileorfoldername.path)
                listfiles(fileorfoldername)
            else:
                filelist.append(fileorfoldername.path)
    listfiles(inputdir)
    return filelist

def labelsfromfilepaths(inputlist):
    labellist = list()
    for filepath in inputlist:
        foldersofpath = filepath.split("\\")
        filename = foldersofpath[-1]                        #grab last part of the file path which is always the target file
        label = filename.split("_")[0]                      #grab only the species out of the filename, only Platypleuras seems to have problematic inconsistent syntax
        labellist.append(label)
    return labellist

X_train = getfilepaths(ROOT_PATH + "\insectsound_MariusFaißMSc\ManualTrain")
X_test = getfilepaths(ROOT_PATH + "\insectsound_MariusFaißMSc\ManualTest")
X_val = getfilepaths(ROOT_PATH + "\insectsound_MariusFaißMSc\ManualValidation")

y_train = labelsfromfilepaths(X_train)
y_test = labelsfromfilepaths(X_test)
y_val = labelsfromfilepaths(X_val)

###AUDIO TESTING PART
import math, random
import torch        #pip install torch
import torchaudio   #pip install torchaudio
from torchaudio import transforms
from IPython.display import Audio       #pip install IPython
#for no audio backend error use SoundFile for Windows "pip install PySoundFile", Sox for Linux "pip install sox"

class AudioUtil():
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)
  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)
  def time_shift(aud, shift_limit):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

# for file in X_train:  #confirms 44100 rate on all (training) files
#     print(AudioUtil.open(file))