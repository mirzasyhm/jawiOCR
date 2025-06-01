# model.py
import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        # x shape: (seq_len, batch_size, nIn)
        rec, _ = self.rnn(x)
        # rec shape: (seq_len, batch_size, nHidden * 2)
        T, b, h = rec.size()
        # Reshape for linear layer: (seq_len * batch_size, nHidden * 2)
        out = self.embedding(rec.view(T * b, h))
        # Reshape back: (seq_len, batch_size, nOut)
        return out.view(T, b, -1)

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        """
        CRNN model definition.
        Args:
            imgH (int): Input image height (must be multiple of 16).
            nc (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            nclass (int): Number of output classes (alphabet_size + 1 for CTC blank).
            nh (int): Number of hidden units in the BiLSTM layers.
            leakyRelu (bool): Whether to use LeakyReLU instead of ReLU (default: False).
        """
        super().__init__()
        if imgH % 16 != 0:
            raise ValueError(f"imgH ({imgH}) must be a multiple of 16")

        ks = [3, 3, 3, 3, 3, 3, 2]  # Kernel sizes
        ps = [1, 1, 1, 1, 1, 1, 0]  # Paddings
        ss = [1, 1, 1, 1, 1, 1, 1]  # Strides
        nm = [64, 128, 256, 256, 512, 512, 512] # Number of feature maps

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(True))

        # CNN Part
        # convRelu(0)                        # Output H = imgH, W = imgW
        # cnn.add_module('pooling0', nn.MaxPool2d(2, 2)) # Output H = imgH/2, W = imgW/2
        # convRelu(1)
        # cnn.add_module('pooling1', nn.MaxPool2d(2, 2)) # Output H = imgH/4, W = imgW/4
        # convRelu(2, True)
        # convRelu(3)
        # cnn.add_module('pooling2', nn.MaxPool2d((2, 2), (2, 1), (0, 1))) # Output H = imgH/8, W = ~imgW/4
        # convRelu(4, True)
        # convRelu(5)
        # cnn.add_module('pooling3', nn.MaxPool2d((2, 2), (2, 1), (0, 1))) # Output H = imgH/16, W = ~imgW/4
        # convRelu(6, True)                  # Output H = 1 (since imgH/16 = 1), W = ~imgW/4 - 1
        
        # Using the exact structure from Cell 9
        cnn.add_module('conv0', nn.Conv2d(nc, 64, 3, 1, 1))
        cnn.add_module('relu0', nn.ReLU(True))
        cnn.add_module('pool0', nn.MaxPool2d(2, 2)) # 64x H/2 x W/2

        cnn.add_module('conv1', nn.Conv2d(64, 128, 3, 1, 1))
        cnn.add_module('relu1', nn.ReLU(True))
        cnn.add_module('pool1', nn.MaxPool2d(2, 2)) # 128 x H/4 x W/4

        cnn.add_module('conv2', nn.Conv2d(128, 256, 3, 1, 1))
        cnn.add_module('relu2', nn.ReLU(True)) # Original cell does not have BN here
        
        cnn.add_module('conv3', nn.Conv2d(256, 256, 3, 1, 1))
        cnn.add_module('batchnorm3', nn.BatchNorm2d(256))
        cnn.add_module('relu3', nn.ReLU(True))
        cnn.add_module('pool2', nn.MaxPool2d((2, 2), (2, 1), (0, 1))) # 256 x H/8 x W/4 effectively

        cnn.add_module('conv4', nn.Conv2d(256, 512, 3, 1, 1))
        cnn.add_module('batchnorm4', nn.BatchNorm2d(512))
        cnn.add_module('relu4', nn.ReLU(True))
        
        cnn.add_module('conv5', nn.Conv2d(512, 512, 3, 1, 1))
        cnn.add_module('batchnorm5', nn.BatchNorm2d(512))
        cnn.add_module('relu5', nn.ReLU(True))
        cnn.add_module('pool3', nn.MaxPool2d((2, 2), (2, 1), (0, 1))) # 512 x H/16 x W/4 effectively

        cnn.add_module('conv6', nn.Conv2d(512, 512, 2, 1, 0)) # Kernel 2x1, Stride 1, Padding 0
        # This conv requires input height to be 1 after pooling3.
        # If imgH=32, H/16 = 2. So pool3 output is 2xW'.
        # The conv6 with kernel (2,x) (implicitly (2,2) if input is H=2, W=W') becomes (512, 512, 1, W'')
        # So this layer ensures h=1.
        cnn.add_module('relu6', nn.ReLU(True))

        self.cnn = cnn

        # RNN Part
        # The input features to RNN are of size 512 (output of CNN)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh), # nIn for first BiLSTM is num_channels from CNN
            BidirectionalLSTM(nh, nh, nclass) # nOut for second BiLSTM is num_classes
        )

    def forward(self, x):
        # x shape: (batch_size, nc, imgH, imgW)
        conv_features = self.cnn(x)
        # conv_features shape: (batch_size, 512, 1, W_sequence)
        
        b, c, h, w = conv_features.size()
        if h != 1:
            # This can happen if imgH is not perfectly reduced to 1 by the CNN.
            # For imgH=32, the CNN structure is designed to output h=1.
            # (32 -> 16 -> 8 -> 4 -> 2 -> 1)
            # If the input image height to conv6 is not 2, this assertion might fail.
            # E.g. if pool3 output height is H_pool3, conv6 with kernel (2,1,0) if H_pool3=2 -> H_out=1
            # We need to ensure that h=1 after the CNN.
            # For example, an adaptive pooling layer could be used if imgH varies.
            # For a fixed imgH=32, this should be h=1.
            raise AssertionError(f"Expected height of CNN output to be 1, but got {h}. "
                                 f"Input imgH was {self.cnn_input_height_tracker if hasattr(self, 'cnn_input_height_tracker') else 'unknown'}")

        # Prepare features for RNN: (seq_len, batch_size, num_features)
        # Here, seq_len is W_sequence, num_features is 512 (channels)
        conv_features = conv_features.squeeze(2)  # Remove height dim: (batch_size, 512, W_sequence)
        conv_features = conv_features.permute(2, 0, 1)  # (W_sequence, batch_size, 512)
        
        rnn_output = self.rnn(conv_features)
        # rnn_output shape: (W_sequence, batch_size, nclass)
        return rnn_output # This output is suitable for CTCLoss (expects T, N, C)
