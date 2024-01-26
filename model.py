import torch.nn.functional as F
import torch.nn as nn

class GreekLettersCNN(nn.Module):
    def __init__(self):
        super(GreekLettersCNN, self).__init__()
        self.batch_norm = nn.BatchNorm2d(1) # 1 x 14 x 14
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2) ## 64 x 14 x 14
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1) ## 32 x 14 x 14
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1) ## same
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 14 * 14, 24)

    def forward(self, x):
        # Continue with the rest of the model architecture
        x = self.batch_norm(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    