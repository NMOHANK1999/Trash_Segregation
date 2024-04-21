import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import wandb
import pdb



#Get the paths/labels. This one includes an excpetion incase one fails.
def prepare_data(data_dir):
    categories = ['Trash', 'Plastic', 'Paper', 'Metal', 'Glass', 'Cardboard']
    image_paths = []
    labels = []  # Numerical labels: 0 for Trash, 1 for Plastic, etc.

    for label, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        try:
            for file in os.listdir(category_dir):
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_paths.append(os.path.join(category_dir, file))
                    labels.append(label)
        except Exception as e:
            print(f"Failed to process category {category}: {e}")
            continue

    return image_paths, labels

folder_path = "data"


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # Load the image as a PIL Image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label
    
    
#with all the transforms
training_transform_all = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.RandomCrop(224),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.1),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Without Random Crop
training_transform_wo_randomcrop = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.1),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Without Random Rotation
training_transform_wo_rotation = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(0.1),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#without Horzontal Flip
training_transform_wo_horizontalflip = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.RandomCrop(224),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#Without Colour jitter
training_transform_wo_coljitter = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.RandomCrop(224),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

###############################################study this##################################3
#without Sharpness adjust
training_transform_wo_adjustsharpness = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.RandomCrop(224),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.1),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#transforms for test and val
val_transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#getting the paths
image_paths, labels = prepare_data(folder_path)

#getting the train, test and val paths
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.20, random_state=0, stratify=labels) #This time we split the oringinal to train/test, so we use the stratify=labels


def train(model = None,
          batch_size = None,
          lr = None,
          step_size = None,
          gamma = None,
          N_epochs = None,
          transform_type = None,
):
    #make the changes
    if transform_type == "all":
        train_dataset = CustomDataset(train_paths, train_labels, transform=training_transform_all)
    elif transform_type == "Random_Crop":
        train_dataset = CustomDataset(train_paths, train_labels, transform=training_transform_wo_randomcrop)
    elif transform_type == "Random_Rotation":
        train_dataset = CustomDataset(train_paths, train_labels, transform=training_transform_wo_rotation)
    elif transform_type == "Horzontal_Flip":
        train_dataset = CustomDataset(train_paths, train_labels, transform=training_transform_wo_horizontalflip)
    elif transform_type == "Colour_jitter":
        train_dataset = CustomDataset(train_paths, train_labels, transform=training_transform_wo_coljitter)
    elif transform_type == "Sharpness_adjust":
        train_dataset = CustomDataset(train_paths, train_labels, transform=training_transform_wo_adjustsharpness)

    val_dataset = CustomDataset(val_paths, val_labels, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)


    loss_fun = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(params= model.parameters(), lr = lr)
    lr_schedule = lr_scheduler.StepLR(optimizer= optimizer, step_size = step_size, gamma = gamma)

    training_losses = []
    training_acces = []
    val_losses = []
    val_acces = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in tqdm(range(N_epochs)):
        train_loss = 0.0
        train_acc = 0.0
        model.train()

        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            _, predicted_classes = torch.max(pred, 1)
            correct_predictions = (predicted_classes == y).float()

            loss = loss_fun(pred, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += correct_predictions.sum().item() / y.size(0)


        # Average loss and accuracy for the epoch
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        # Validation phase
        val_loss = 0.0
        val_acc = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                _, predicted_classes = torch.max(pred, 1)
                correct_predictions = (predicted_classes == y).float()

                loss = loss_fun(pred, y.long())  # Ensure consistent data type
                val_loss += loss.item()
                val_acc += correct_predictions.sum().item() / y.size(0)


        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
        wandb.log({"Train_Loss": train_loss, "Train_Accuracy": train_acc, "Validation_Loss": val_loss, "Validation_Accuracy": val_acc})
        lr_schedule.step() 

        print(f"Epoch {epoch+1}/{N_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return val_acc


def obj(config):

    #makes sure all initializations are the same, cuda uses all deterministic functions and benchmarking(most efficient operations) is off
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

    model = torchvision.models.resnet50().cuda()
    model.load_state_dict(torch.load('resnet50-11ad3fa6.pth'))

    for param in model.parameters():
        param.requires_grad = config.freeze_model

    in_feat = model.fc.in_features # 2048 input features
    classes = 6 # output feature classes.

    model.fc = nn.Sequential(
        nn.Linear(in_feat, 512),
        nn.ReLU(),
        nn.Linear(512, classes)
    )

    val_acc = train (
            model= model,
            batch_size = config.batch_size ,
            lr = config.lr,
            step_size = config.step_size,
            gamma = config.gamma,
            N_epochs = config.N_epochs,
            transform_type = config.transform_type,)
    
    return val_acc

def start_train():
    wandb.init()
    score = obj(wandb.config)
    wandb.log({"score": score})


if __name__ == "__main__":
  sweep_config = {
      "method" : "grid",
      "metric": {
          "name" : "accuracy",
          "goal" : "maximize"
      },
      "parameters": {
          "batch_size" : {
              "values": [32]
          },
          "lr" : {
              "values": [1e-5]
          },
          "step_size" : {
              "values": [20]
          },
          "gamma" : {
              "values": [0.9]
          },
          "N_epochs" : {
              "values": [10]
          },
          "transform_type" : {
              "values": ["all", "Random_Crop", "Random_Rotation", "Horzontal_Flip", "Colour_jitter", "Sharpness_adjust"]
          },
          "freeze_model" : {
              "values": [True]
          },
      }
  }
  sweep_id = wandb.sweep(sweep_config, project = "ML_AI_Project")
  wandb.agent(sweep_id, function = start_train)
