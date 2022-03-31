import numpy as np
from PIL import Image
import librosa
import json

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

    nbrFrame = len(input)

    # Liste des shape keys
    sk_list = ["Basis", "jaw_open", "left_eye_closed", "mouth_open", "right_eye_closed", "smile", "smile_left", "smile_right"]

    # Création d'un faux output pour tester le code avant d'avoir le modèle
    output = []
    for i in range(nbrFrame):
        output.append([1, 1, 1, 1, 1, 1, 1, 1])
    
    basis = np.loadtxt('../AlphData/shape_keys_v0/Basis.txt')
    jaw_open = np.loadtxt('../AlphData/shape_keys_v0/jaw_open.txt')
    left_eye_closed = np.loadtxt('../AlphData/shape_keys_v0/left_eye_closed.txt')
    mouth_open = np.loadtxt('../AlphData/shape_keys_v0/mouth_open.txt')
    right_eye_closed = np.loadtxt('../AlphData/shape_keys_v0/right_eye_closed.txt')
    smile_left = np.loadtxt('../AlphData/shape_keys_v0/smile_left.txt')
    smile_right = np.loadtxt('../AlphData/shape_keys_v0/smile_right.txt')
    smile = np.loadtxt('../AlphData/shape_keys_v0/smile.txt')

    response = [
        {
            "frame": k,
            "mesh": json.dumps((output[i][0] * basis + output[i][1] * jaw_open + output[i][2] * left_eye_closed + output[i][3] * mouth_open + output[i][4] * right_eye_closed+ output[i][5] * smile + output[i][6] * smile_left + output[i][7] * smile_right).tolist())
        } for k in range(nbrFrame)
        
    ]
    return response