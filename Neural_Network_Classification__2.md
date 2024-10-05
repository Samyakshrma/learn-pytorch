
![Screenshot 2024-10-05 082230](https://github.com/user-attachments/assets/bac62a0c-50c5-4fc0-b41c-170f55b8c493)

## Turn Data to tensor
```
# Turn data into tensors
# Otherwise this causes issues with computations later on
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
```
## Test train split
```
# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

len(X_train), len(X_test), len(y_train), len(y_test)
```
## Cpu default gpu if available
```
# Standard PyTorch imports
import torch
from torch import nn

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```
## Example neural network structure
```
class circleModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(in_features=2, out_features=5)
    self.layer2 = nn.Linear(in_features=5, out_features=1)
  
  def forward(self,x):
    return self.layer2(self.layer1(x))
```
`nn.Linear` is basic linear regression that looks like this y = x*W + b
