import os
import librosa
import copy
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

qint8_available = False

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
valid_dataloader = DataLoader(valid_dataset, batch_size=1)
valid_dataloader32 = DataLoader(valid_dataset, batch_size=32)
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

## device has to be cuda to do AMP
if torch.cuda.is_available():
    device = "cuda"
else:
    raise Exception("No CUDA device found. This work must be run on CUDA device.")

# load model
model = SpectrogramCNN(num_classes=6)
model.load_state_dict(torch.load('checkpoint/baseline_spectrogram_cnn_model.pth'))
model.eval().to(device)

if qint8_available:
    qmodel = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

def validation(input_model, AMP = False):
  correct_predictions = 0
  total_samples = 0
  inference_latency_list = []
  inference_latency_list_b1 = []
  with torch.no_grad():
      
      for inputs, labels in valid_dataloader:
          inputs = inputs.to(device)
          labels = labels.to(device)
          i_start_time = time.time()
          if AMP:
            with torch.cuda.amp.autocast():
              outputs = model(inputs)
          else:
            outputs = model(inputs)
          _, predicted = torch.max(outputs, 1)
          total_samples += labels.size(0)
          correct_predictions += (predicted == labels).sum().item()
          i_end_time = time.time()
          inference_latency_list_b1 += [(i_end_time-i_start_time)/labels.size(0)] * labels.size(0)


      for inputs, labels in valid_dataloader32:
          inputs = inputs.to(device)
          labels = labels.to(device)
          i_start_time = time.time()
          if AMP:
            with torch.cuda.amp.autocast():
              outputs = model(inputs)
          else:
            outputs = model(inputs)
          _, predicted = torch.max(outputs, 1)
          total_samples += labels.size(0)
          correct_predictions += (predicted == labels).sum().item()
          i_end_time = time.time()
          inference_latency_list += [(i_end_time-i_start_time)/labels.size(0)] * labels.size(0)

  # Throw away first iteration
  inference_latency_list = inference_latency_list[64:]
  inference_latency_list_b1 = inference_latency_list_b1[1:]

  accuracy = correct_predictions / total_samples * 100
  print(f'Validation Accuracy: {accuracy:.2f}%')
  print(f'Average Inference Latency: {np.mean(inference_latency_list)/64}')
  print(f'Average Inference Latency(b1): {np.mean(inference_latency_list_b1)}')

validation(model)
validation(model, AMP = True)

if qint8_available:
    f=print_size_of_model(model,"fp32")
    q=print_size_of_model(qmodel,"qint8")
    print("{0:.2f} times smaller".format(f/q))
