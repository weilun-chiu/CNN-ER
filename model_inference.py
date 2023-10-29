import os
import librosa

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

valid_dir = './Valid'

emotion2idx_dict = {'ANG': 0, 'NEU': 1, 'DIS': 2, 'SAD': 3, 'FEA': 4, 'HAP': 5}
idx2emotion_dict = {0: 'ANG', 1: 'NEU', 2: 'DIS', 3: 'SAD', 4: 'FEA', 5: 'HAP'}

# find the longest spectrogram length in the training dataset
spectrogram_length = 495
feature_size = 513

class Speech_Dataset(Dataset):
    def __init__(self, data_dir, emotion2idx_dict, spectrogram_length):
        self.data_dir = data_dir
        self.emotion2idx_dict = emotion2idx_dict
        self.spectrogram_length = spectrogram_length

        self.filenames = os.listdir(data_dir)
        self.labels = [self.emotion2idx_dict[filename.split('_')[2]] for filename in self.filenames]

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels[idx]

        # load the speech file and calculate spectrogram
        speech_audio, _ = librosa.load(os.path.join(self.data_dir, filename), sr = 16000)
        spectrogram = librosa.stft(speech_audio, n_fft=1024, hop_length=160, center=False, win_length=1024)
        spectrogram = abs(spectrogram)
        
        feature_size, length = spectrogram.shape

        # modify the length of the spectrogram to be the same as the specified length
        if length > self.spectrogram_length:
            spectrogram = spectrogram[:, :self.spectrogram_length]
        else:
            cols_needed = self.spectrogram_length - length
            spectrogram = np.concatenate((spectrogram, np.zeros((feature_size, cols_needed))), axis=1)
        
        return np.expand_dims(spectrogram.astype(np.float32), axis=0) , label

valid_dataset = Speech_Dataset(valid_dir, emotion2idx_dict, spectrogram_length)
valid_dataloader = DataLoader(valid_dataset, batch_size=32)

# three layers: baseline
# Training Accuracy: 78.81%
# Validation Accuracy: 50.11%

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 64 * 61, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 16 * 64 * 61)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# load model
model = SpectrogramCNN(num_classes=6)
model.load_state_dict(torch.load('checkpoint/baseline_spectrogram_cnn_model.pth'))
model.eval()

# validation script
correct_predictions = 0
total_samples = 0
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in valid_dataloader:
        export_output = torch.onnx.dynamo_export(model, inputs)
        export_output.save("cnn.onnx")
        exit()
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())

accuracy = correct_predictions / total_samples * 100
print(f'Validation Accuracy: {accuracy:.2f}%')
