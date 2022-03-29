import numpy as np
from PIL import Image
import librosa

import torch


def load_model():
    """
    Loads and returns the pretrained model
    """
    model = torch.load('C:/Users/Enzo.Magal/Documents/Enzo2021/models/sk_model.pth')
    print("Model loaded")
    return model


def prepare_audio(audio, sample_rate, target):
    sample = []
    for i in range(target):
        sample.append(audio[int(i*(len(audio)/target)):int((i+1)*(len(audio)/target))])
    mel_spec = []
    for i in range(target):
        spectrogram = librosa.stft(sample[i], n_fft = 1024, hop_length = 41, win_length=82)
        sgram_mag, _ = librosa.magphase(spectrogram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr= sample_rate)
        mel_spec.append(librosa.amplitude_to_db(mel_scale_sgram, ref=np.min))

    return mel_spec


def predict(input, model):

    sk_list = ["Basis", "jaw_open", "left_eye_closed", "mouth_open", "right_eye_closed", "smile", "smile_left", "smile_right"]
    
    response = [
        {"shape key": sk, "weight": 1} for sk in sk_list
    ]
    return response