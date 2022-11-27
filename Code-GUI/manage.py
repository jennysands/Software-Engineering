#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

### imports for image classifier ###
import matplotlib.image as mpimg
import pandas as pd
import os
import torch
import torchaudio
from torchaudio import transforms
from torchaudio.utils import download_asset
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import numpy as np
### imports for image classifier ###

### EXTRA imports for audio classifier ###
from torch.utils.data import random_split
import soundfile as sf
### EXTRA imports for audio classifier ###


### BEGIN OF CLASSIFIERS ###

### IMAGE CLASSIFIER ###

class imgdata(Dataset):
    def __init__(self, df, data_path, transform = None):
        self.df = df
        self.transform = transform
        self.data_path = data_path
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        img_file = self.df.loc[idx, 'Path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'ID']
        img = mpimg.imread(img_file)
        if self.transform is not None:
            img = self.transform(img)
        return img, class_id
    
    
class imgClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=450)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x
    
    def preprocess(model, path_to_image, path_to_dummy):
        
        MEAN = np.array([0.485, 0.456, 0.406])
        STD = np.array([0.229, 0.224, 0.225])
        
        IMG_path = [path_to_image, path_to_dummy]
        df_test = pd.DataFrame( {"Path":IMG_path, "ID":[0, 1]} )
        
        test_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(MEAN,STD), transforms.Resize((224, 224))])  

        testDS = imgdata(df_test, IMG_path, test_transform)
        test_dl = torch.utils.data.DataLoader(testDS, batch_size=64, shuffle=False)
        
        return test_dl
    
    def predict(model, preprocessed_image):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        correct_prediction = 0
        total_prediction = 0

        # Disable gradient updates
        with torch.no_grad():
            for data in preprocessed_image:
                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(device), data[1].to(device)

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s
                
                outputs = model(inputs)
                # print(outputs)

                    # Get the predicted class with the highest score
                _, prediction = torch.max(outputs,1)
                    # Count of predictions that matched the target label
                return int(prediction[0])
            
### END OF IMAGE CLASSIFIER ###
        
### BEGIN OF AUDIO CLASSIFIER ###

#Creating the main class, which will have necessary methods for preprocessing

class AudioUtil():
    
    #loading and reading the files, I will probably have to look far a different loading method for .flac files
   
    @staticmethod
    def file_open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return sig, sr
    
    #Creating two channels

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
        # Nothing to do
            return aud

        if (new_channel == 1):
        # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
        # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

            return ((resig, sr))

    # Resampling
    
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
        # Nothing to do
            return aud

        num_channels = sig.shape[0]
      # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
        # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))
    
    #Equalizing the sample sizes
    
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
      
        return (sig, sr)
    
    #Shifting time. (For now, I don't know what I should add in the second parameter when testing)
    
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)
    
    # Mel spectrogram
    
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
    
    # Augemented Mel spectrogram
    
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
    
class SoundDS(Dataset):
    def __init__(self, df):
        self.df = df
        #self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4
            
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
    def __len__(self):
        return len(self.df)    
    
  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
    def __getitem__(self, idx):
    # Absolute file path of the audio file - concatenate the audio directory with
    # the relative path
        audio_file = self.df.loc[idx, 'Path']
    # Get the Class ID
        class_id = self.df.loc[idx, 'ID']

        aud = AudioUtil.file_open(audio_file)
        # Some sounds might have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        aug_sgram = aug_sgram.squeeze(0)
        return aug_sgram, class_id

class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=274)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x
    
    def preprocess(model, path_to_audiofile, path_to_dummy_audiofile):
        
        path_test_bird_audio = [path_to_audiofile, path_to_dummy_audiofile]
        birds_audio_test_df =  pd.DataFrame({"Path":path_test_bird_audio, "ID":[0,1]})
        
        test_DS = SoundDS(birds_audio_test_df)
        
        # Create training and validation data loaders
        test_dl = torch.utils.data.DataLoader(test_DS, batch_size=64, shuffle=False)
        
        return test_dl
        
    def predict(model, preprocessed_audio_file):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            for data in preprocessed_audio_file:
        # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

        # Get predictions
                outputs = model(inputs)

        # Get the predicted class with the highest score
                _, prediction = torch.max(outputs,1)
                return int(prediction[0])
        
        
            
### END OF AUDIO CLASSIFIER ###
### End of Classifiers ###






def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'se_project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
