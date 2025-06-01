# model.py
import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        rec, _ = self.rnn(x)
        T, b, h = rec.size()
        out = self.embedding(rec.view(T * b, h))
        return out.view(T, b, -1)

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super().__init__()
        assert imgH % 16 == 0, f"imgH ({imgH}) must be a multiple of 16"
        
        # CNN part: Exactly as in your Cell 9
        # This will create layers named cnn.0, cnn.1, cnn.2, etc.
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(True), 
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x H/2 x W/2

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128 x H/4 x W/4

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(True), # 256 x H/4 x W/4
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,1), padding=(0,1)), # Output: 256 x H/8 x W/4 (approx)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512), nn.ReLU(True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,1), padding=(0,1)), # Output: 512 x H/16 x W/4 (approx)
            
            # This Conv2d layer makes the height dimension 1.
            # If imgH=32, H/16 = 2. So input to this conv is H_feat=2.
            # Kernel (height) = 2 makes output height 1.
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0), nn.ReLU(True) # Output: 512 x 1 x W_seq
        )
        
        # RNN Part
        # Input features to RNN from CNN will be 512 channels
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, x):
        # x shape: (batch, nc, imgH, imgW)
        conv_features = self.cnn(x)
        # conv_features shape: (batch, 512, 1, W_sequence)
        
        b, c, h, w = conv_features.size()
        assert h == 1, f"Expected height of CNN output to be 1, but got {h}. Input imgH might not be processed correctly."
        
        # Prepare features for RNN: (seq_len, batch, num_features)
        # Here, seq_len is W_sequence, num_features is 512 (channels)
        conv_features = conv_features.squeeze(2)  # Remove height dim: (batch, 512, W_sequence)
        conv_features = conv_features.permute(2, 0, 1)  # (W_sequence, batch, 512)
        
        rnn_output = self.rnn(conv_features)
        # rnn_output shape: (W_sequence, batch, nclass)
        return rnn_output # This output is suitable for CTCLoss
