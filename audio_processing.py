import os
import torch
import torchaudio
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rechannel(aud, new_ch):
    sig, sr = aud
    if (sig.shape[0] == new_ch):
        return sig, sr
    if (new_ch == 1):
        resig = sig[:1, :]
    else:
        resig = torch.cat([sig, sig])
    return (resig, sr)

def resample(aud, new_sr):
    sig, sr = aud
    if (sr == new_sr):
        return sig, sr
    resampled_ch1 = torchaudio.transforms.Resample(sr, new_sr)(sig[:1,:])
    resampled_ch2 = torchaudio.transforms.Resample(sr, new_sr)(sig[1:,:])
    resampled_sig = torch.cat([resampled_ch1, resampled_ch2])
    return (resampled_sig, new_sr)

def resize(aud, max_ms):
    sig, sr = aud
    num_ch, sig_len = sig.shape
    max_len = sr // 1000 * max_ms
    if sig_len > max_len:
        sig = sig[:,:max_len]
    elif sig_len < max_len:
        pad_begin_len = np.random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len
        pad_begin = torch.zeros((num_ch, pad_begin_len))
        pad_end = torch.zeros((num_ch, pad_end_len))
        sig = torch.cat((pad_begin, sig, pad_end), 1)
    return (sig, sr)

def time_shift(aud, shift_limit):
    sig, sr = aud
    sig_len = sig.shape[1]
    shift_amt = int(np.random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

def mel_spectrogram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig, sr = aud
    top_db = 80
    # shape [channel, n_mels, time]
    spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(sig)
    spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(spec)
    return spec

def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec
    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
    return aug_spec


class Inference:

    def __init__(self, model, rate, channels, duration, shift_pct):
        self.model = model
        self.rate = rate
        self.channels = channels
        self.duration = duration
        self.shift_pct = shift_pct
        self.model.eval()
    
    def preprocess(self, path):
        raw_aud = torchaudio.load(path)
        resr_aud = resample(raw_aud, self.rate)
        rech_aud = rechannel(resr_aud, self.channels)
        resz_aud = resize(rech_aud, self.duration)
        shft_aud = time_shift(resz_aud, self.shift_pct)
        raw_spec = mel_spectrogram(shft_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_spec = spectro_augment(raw_spec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        return aug_spec
    
    def predict(self, spectrum):
        with torch.no_grad():
            output = self.model(input.unsqueeze(0).to(device))
            # print(f'Prediction: {output}')
            _, prediction = torch.max(output, 1)
            # print(f'Prediction: {prediction}')
            return prediction.item()