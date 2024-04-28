import torch
from model import ANNClassifier
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



class FGSM:
  def __init__(self, model, criterion, epsilon=0.3):
    self.model = model
    self.criterion = criterion
    self.epsilon = epsilon
    
  def fgsm_attack(self, image, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + self.epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
  
  def denorm(self, batch, mean=[0.1307], std=[0.3081]):
    device = "cpu"
    if isinstance(mean, list):
      mean = torch.tensor(mean).to(device)
      
    if isinstance(std, list):
      std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

  def apply(self):
    """
    Perform the Fast Gradient Sign Method attack on the model and return the results.
    The result should be such that it contains evasion rate, adversarial examples, and
    all necessary information to plot and answer the questions.
    return: dict
    """
    self.model.eval()
    correct = 0
    results = {"adv_examples": [], "labels": [], "classified_as": [], "examples": []}
    device = "cpu"
    
    total_examples = len(self.model.test_loader.dataset)

    # Specify the number of examples to sample (e.g., 1000)
    num_samples = 1000

    # Create indices to sample from (randomly shuffle indices)
    indices = torch.randperm(total_examples)

    # Take the first 'num_samples' indices for sampling
    subset_indices = indices[:num_samples]

    # Create a SubsetRandomSampler using the selected subset indices
    subset_sampler = SubsetRandomSampler(subset_indices)

    # Create a DataLoader for the subset of examples
    subset_loader = DataLoader(self.model.test_loader.dataset,
                              batch_size=self.model.test_loader.batch_size,
                              sampler=subset_sampler)

    # Loop over all examples in test set
    for data, target in subset_loader:
      # Send the data and label to the device
      data, target = data.to(device), target.to(device)
      
      # Set requires_grad attribute of tensor. Important for Attack
      data.requires_grad = True

      # Forward pass the data through the model
      output = self.model(data)
      # init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

      # If the initial prediction is wrong, don't bother attacking, just move on
      # if init_pred.item() != target.item():
      #   continue

      # Calculate the loss
      loss = self.criterion(output, target)

      # Zero all existing gradients
      self.model.zero_grad()

      # Calculate gradients of model in backward pass
      loss.backward()

      # Collect ``datagrad``
      data_grad = data.grad.data

      # Restore the data to its original scale
      data_denorm = self.denorm(data)

      # Call FGSM Attack
      perturbed_data = self.fgsm_attack(data_denorm, data_grad)

      # Reapply normalization
      perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

      # Re-classify the perturbed image
      output = model(perturbed_data_normalized)

      # Check for success
      final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
      if final_pred.item() == target.item():
        correct += 1
      
      adv_ex = perturbed_data.detach().cpu() 
      adv_ex = F.interpolate(adv_ex, size=(28, 28), mode='bilinear', align_corners=False)
      data_ex = data.detach().cpu()
      data_ex = F.interpolate(data_ex, size=(28, 28), mode='bilinear', align_corners=False)
      
      results["examples"].append(data_ex)
      results["adv_examples"].append(adv_ex)
      results["labels"].append(target.item())
      results["classified_as"].append(final_pred.item())

    # Calculate final accuracy for this epsilon
    final_acc = 100 * correct / float(len(subset_loader))
    results["accuracy"] = final_acc
    results["evasion_rate"] = 100 - final_acc
    # print(f"Test Accuracy = {correct} / {len(subset_loader)} = {final_acc}%")

    # Return the accuracy and an adversarial example
    return results
  
  def plot(self, results):
    num_digits = 10
    digit_indices = [np.where(np.array(results["labels"]) == i)[0][0] for i in range(num_digits)]
    
    plt.figure(figsize=(30, 10))
    for i in range(num_digits):
        # Original image
        plt.subplot(3, num_digits, i + 1)
        plt.imshow(results["examples"][digit_indices[i]].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        
        # Adversarial example
        plt.subplot(3, num_digits, num_digits + i + 1)
        plt.imshow(results["adv_examples"][digit_indices[i]].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        
        # L2 norm of adversarial noise
        noise = results["adv_examples"][digit_indices[i]] - results["examples"][digit_indices[i]]
        noise_norm = torch.norm(noise).item()
        plt.subplot(3, num_digits, 2*num_digits + i + 1)
        # plt.title(f"L2 Norm: {noise_norm:.2f}")
        plt.imshow(noise.squeeze().detach().cpu().numpy(), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("attack.png")

if __name__ == "__main__":

  attack_params = {
    "batch_size": 1,
    "epsilon": 0.1,
    "learning_rate": 0.01,
    "model_name": "model.pth"
  }

  # Load the trained model
  model = ANNClassifier(attack_params)
  model.load_state_dict(torch.load(attack_params["model_name"]))

  # Load the test data
  train_loader, test_loader = model.get_dataloaders()
  model.train_loader = train_loader
  model.test_loader = test_loader

  ## Attack ##

  # Attack the model on test set
  attack = FGSM(model, torch.nn.CrossEntropyLoss(), attack_params["epsilon"])
  results: dict = attack.apply()
  
  print(f"Accuracy = {results['accuracy']}%")
  print(f"Evasion Rate = {results['evasion_rate']}%")
  
  # Assuming we have collected results in results["labels"] and results["classified_as"]
  true_labels = np.array(results["labels"])
  predicted_labels = np.array(results["classified_as"])

  # Compute confusion matrix
  cm = confusion_matrix(true_labels, predicted_labels)

  # Display the confusion matrix
  print("Confusion Matrix:")
  print(cm)
  
  # Plot
  attack.plot(results)

  # check if the results contains a key named "adv_examples" and check each element
  # is a torch.Tensor. THIS IS IMPORTANT TO PASS THE TEST CASES!!!
  assert "adv_examples" in results.keys(), "Results should contain a key named 'adv_examples'"
  assert all([isinstance(x, torch.Tensor) for x in results["adv_examples"]]), "All elements in 'adv_examples' should be torch.Tensor"

  # check the image size should be 1x28x28
  assert results["adv_examples"][0].shape[1] == 1, "The image should be grayscale"
  assert results["adv_examples"][0].shape[2] == 28, "The image should be 28x28"
