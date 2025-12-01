from torch.utils.data import Dataset
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as  F


class CIFAR10_dataset(Dataset):

    def __init__(self, partition = "train", transform = None):

        print("\nLoading CIFAR10 ", partition, " Dataset...")
        self.partition = partition
        self.transform = transform
        if self.partition == "train":
            self.data = torchvision.datasets.CIFAR10('.data/',
                                                     train=True,
                                                     download=True)
        else:
            self.data = torchvision.datasets.CIFAR10('.data/',
                                                     train=False,
                                                     download=True)
        print("\tTotal Len.: ", len(self.data), "\n", 50*"-")

    def from_pil_to_tensor(self, image):
        return torchvision.transforms.ToTensor()(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Image
        image = self.data[idx][0]
        image_tensor = self.transform(image)

        # Label
        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()

        return {"img": image_tensor, "label": label}
    


class ICCNN(nn.Module):
    """
    ICCNN: A deep convolutional neural network for CIFAR-10 (3x32x32 images).

    High-level structure:
    - 3 convolutional blocks (Conv-Conv-Conv-MaxPool) with increasing channels:
        Block 1:  64  channels, output 64 x 16 x 16
        Block 2: 256  channels, output 256 x 8 x 8
        Block 3: 512  channels, output 512 x 4 x 4
    - Fully connected classifier:
        8192 -> 8192 -> 4096 -> 10 classes
    - Uses BatchNorm, ReLU, MaxPool and Dropout2d for regularization.
    """
    def __init__(self, num_classes=10):
        super(ICCNN, self).__init__()

        # Block 1: C: 3 -> 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32 * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
        # Block 2: C: 128 -> 256
        self.conv4 = nn.Conv2d(in_channels=64 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        # Block 3: C: 256 -> 512
        self.conv7 = nn.Conv2d(in_channels=128 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
        
        # Batch normalization after the first conv of each block
        self.bn1 = nn.BatchNorm2d(32 * 2)
        self.bn2 = nn.BatchNorm2d(128 * 2)
        self.bn3 = nn.BatchNorm2d(256 * 2)
        
        # Shared layers
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.2)
        self.relu = nn.ReLU()

        # Fully connected classifier
        # --------------------------
        # After 3x MaxPool, spatial sizes are 32x32 -> 16x16 -> 8x8 -> 4x4
        # Channels after last block = 512
        # So flattened feature size = 512 * 4 * 4 = 8192 = 4096 * 2
        self.fc1 = nn.Linear(4096 * 2, 4096 * 2)
        self.fc2 = nn.Linear(4096 * 2, 2048 * 2)
        self.fc3 = nn.Linear(2048 * 2, num_classes)
        

    def forward(self, x):

        # -------- Block 1 --------
        # conv1 + BN + ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        # -------- Block 2 --------
        x = self.relu(self.bn2(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        # -------- Block 3 --------
        x = self.relu(self.bn3(self.conv7(x)))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        # -------- Classifier --------
        # Flatten feature maps to a vector per image
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers with ReLU and dropout
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)