import os
import librosa
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size

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

def quantize_to_int8(float32_tensor):
    # Define min and max values for int8
    min_int8 = -128  # int8 minimum value
    max_int8 = 127   # int8 maximum value
    scale = max_int8 - min_int8
    
    # Calculate min and max values in the input float32 tensor
    float32_tensor = float32_tensor * scale
    min_float32 = float32_tensor.min()
    max_float32 = float32_tensor.max()
    print(f'min={min_float32}, max={max_float32}')

    # Scale and quantize the input float32 tensor
    int8_tensor = ((float32_tensor - min_float32) / (max_float32 - min_float32)) * (max_int8 - min_int8) + min_int8
    int8_tensor = int8_tensor.to(torch.int8)

    print(f'input_dim = {float32_tensor.shape}  output_dim={int8_tensor.shape}')

    return (int8_tensor, min_float32, max_float32)
    
def dequantize_to_float32(int8_tensor, min_float32, max_float32):
    # Calculate the scale factor
    scale_factor = (max_float32 - min_float32) / (int8_tensor.max().item() - int8_tensor.min().item())

    # Dequantize the int8 tensor back to float32
    float32_tensor = (int8_tensor - int8_tensor.min()) * scale_factor + min_float32
    float32_tensor = float32_tensor.to(torch.float32)

    return float32_tensor

class QLinear(nn.Module):
    def __init__(self, linear):
        super(QLinear, self).__init__()
        self.weight, self.weight_min, self.weight_max = quantize_to_int8(linear.weight)
        self.bias, self.bias_min, self.bias_max = quantize_to_int8(linear.bias)

    def forward(self, input):
        # Implement the forward pass using int8 operations
        weight = dequantize_to_float32(self.weight, self.weight_min, self.weight_max)
        bias = dequantize_to_float32(self.bias, self.bias_min, self.bias_max)
        output = torch.nn.functional.linear(input, weight, bias)
        return output


class QSpectrogramCNN(nn.Module):
    def __init__(self, target):
        super(QSpectrogramCNN, self).__init__()
        self.conv1 = target.conv1
        self.bn1 = target.bn1
        self.conv2 = target.conv2
        self.bn2 = target.bn2
        self.conv3 = target.conv3
        self.bn3 = target.bn3
        self.pool = target.pool
        self.fc1 = QLinear(target.fc1)
        self.fc2 = QLinear(target.fc2)
        self.relu = target.relu

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

qmodel = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
# QSpectrogramCNN(model)
# torch.save(model.state_dict(), 'qmodel.pth')


# validation script
correct_predictions = 0
total_samples = 0
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in valid_dataloader:
        outputs = qmodel(inputs)
        
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())

accuracy = correct_predictions / total_samples * 100
print(f'Validation Accuracy: {accuracy:.2f}%')

f=print_size_of_model(model,"fp32")
q=print_size_of_model(qmodel,"int8")
print("{0:.2f} times smaller".format(f/q))
