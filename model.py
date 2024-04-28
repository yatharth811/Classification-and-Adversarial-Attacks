import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from PIL import Image
import os
import matplotlib.pyplot as plt

class CustomDataset(VisionDataset):
  def __init__(self, root, transform=None):
    super(CustomDataset, self).__init__(root, transform=transform)
    self.data = []
    self.targets = []
    self.transform = transform

    # Read images from folders
    for image_file in os.listdir(root):
      image_path = os.path.join(root, image_file)
      image = Image.open(image_path).convert('L')
      label = image_file[-5]
      self.data.append(image)
      self.targets.append(int(label))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    image = self.data[index]
    target = self.targets[index]

    if self.transform:
      image = self.transform(image)

    return image, target


# Define the model architecture
class ANNClassifier(nn.Module):
  def __init__(self, train_params: dict):
    super(ANNClassifier, self).__init__()
    self.train_params = train_params
    self._define_model()
    self.criterion = self._define_criterion()
    self.optimizer = self._define_optimizer()

  def _define_model(self) -> None:
    """
    Define the model architecture
    return: None
    """
    self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                    nn.BatchNorm2d(6),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                  )
    self.layer2 = nn.Sequential(
                    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                  )
    self.fc = nn.Linear(400, 120)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(120, 84)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(84, 10)


  def _define_criterion(self) -> nn.Module:
    """
    Define the criterion (loss function) to use for the model
    return: nn.Module
    """
    return nn.CrossEntropyLoss()

  def _define_optimizer(self) -> torch.optim.Optimizer:
    """
    Define the optimizer to use for the model
    return: torch.optim.Optimizer
    """
    return optim.Adam(self.parameters(), lr=self.train_params['learning_rate'])

  
  def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
    """
    Create the dataloaders(train and test) for the MNIST dataset
    return: DataLoader, DataLoader
    """
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # Create custom datasets
    train_dataset = CustomDataset(root='./train_data', transform=transform)
    test_dataset = CustomDataset(root='./test_data', transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.train_params['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.train_params['batch_size'], shuffle=False)

    return train_loader, test_loader

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    out = self.relu(out)
    out = self.fc1(out)
    out = self.relu1(out)
    out = self.fc2(out)
    return out

  def train_step(self):
    """
    Train and save the model
    return: dict
    """
    self.train()
    epoch_losses = {}
    total_step = len(self.train_loader)
    num_epochs = self.train_params["epochs"]
    device = "cpu"
    for epoch in range(num_epochs):
      epoch_loss = []
      
      for i, (images, labels) in enumerate(self.train_loader):  
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
          
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Store losses for plot
        epoch_loss.append(loss.item())
            
        if (i+1) % 400 == 0:
          print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
      
      epoch_losses[epoch] = epoch_loss
          
    return epoch_losses

  def infer(self):
    """
    Evaluate the model
    return: float
    """
    self.eval()
    device = "cpu"
    with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in self.test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = self.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    accuracy = 100 * correct / total
    # print('Accuracy of the network on the 10000 test images: {} %'.format(accuracy))
    return accuracy
    
  
  def plot_loss(self, results: dict)  -> None:
    """
    Plot the curve loss v/s epochs
    results: dict
    return: None
    """
    
    num_plots = 10  # Number of plots to create
    current_row = 0
    current_plot = 0

    plt.figure(figsize=(18, 12))  # Adjust the figure size as needed

    for epoch in range(num_plots):
      current_row = epoch // 4  # Determine the current row based on the plot index
      current_plot = epoch % 4  # Determine the position within the current row

      plt.subplot(3, 4, current_row * 4 + current_plot + 1)  # Create subplot in the grid
      plt.plot(results[epoch], label='Epoch {}'.format(epoch+1))
      plt.title('Epoch {} Loss'.format(epoch+1))
      plt.xlabel('Iteration')
      plt.ylabel('Loss')
      plt.legend()

    plt.tight_layout()
    plt.savefig("model.png")
    
  def save(self, file_path: str):
    """
    Save the model
    file_path: str
    return: None
    """
    torch.save(self.state_dict(), file_path)


if __name__ == '__main__':
  
  train_params = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 10
  }

  ## Training ##

  # Create the model
  model = ANNClassifier(train_params)

  # Load the data
  train_loader, test_loader = model.get_dataloaders()
  model.train_loader = train_loader
  model.test_loader = test_loader

  # Train the model and return everything you need to report
  # and plot using a dictionary
  results: dict = model.train_step()

  # Plot the loss curve
  model.plot_loss(results)

  # Save the model
  model.save(file_path='model.pth')

  ## Evaluation ##

  # Load the model
  eval_model = ANNClassifier(train_params)
  eval_model.load_state_dict(torch.load('model.pth'))

  # Set the datasets
  train_loader, test_loader = eval_model.get_dataloaders()
  eval_model.train_loader = train_loader
  eval_model.test_loader = test_loader

  # Evaluate the model
  test_accurary = eval_model.infer()
  print(f'Test Accuracy: {test_accurary:.2f}%')
  
  eval_model.test_loader = train_loader
  train_accuracy = eval_model.infer()
  print(f'Train Accuracy: {train_accuracy:.2f}%')