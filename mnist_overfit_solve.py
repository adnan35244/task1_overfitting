import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import copy

# -------------------------------------------------------------------------
# 1. DATA SPLITTING & LOADING
# -------------------------------------------------------------------------
class CustomSubset(Dataset):
    """A custom dataset subset that applies a specific transform."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        img, label = self.subset.dataset[self.subset.indices[index]]
        if self.transform:
            img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.subset)

def get_dataloaders(batch_size=128, use_augmentation=False):
    """
    Creates Train, Validation, and Test dataloaders.
    If use_augmentation is True, applies data augmentation to the training set.
    """
    # Test/Val transform (no augmentation)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Train transform (with optional augmentation to prevent overfitting)
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = base_transform

    # Load base datasets without transforms (we will apply transforms in the subset wrapper)
    # Note: torchvision datasets return PIL Images by default when no transform is provided.
    base_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=base_transform)

    # Split train into train and validation (50,000 train, 10,000 val)
    train_size = 50000
    val_size = 10000
    generator = torch.Generator().manual_seed(42) # For reproducibility
    
    train_subset, val_subset = random_split(base_train_dataset, [train_size, val_size], generator=generator)

    # Wrap subsets to apply transforms
    train_dataset = CustomSubset(train_subset, train_transform)
    val_dataset = CustomSubset(val_subset, base_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# -------------------------------------------------------------------------
# 2. ARCHITECTURE DEFINITION
# -------------------------------------------------------------------------
class NormalMLP(nn.Module):
    """
    A standard, reasonably-sized MLP for MNIST.
    This serves as a baseline before we intentionally induce overfitting.
    We include Dropout here to make it a well-behaved, "normal" model.
    """
    def __init__(self):
        super(NormalMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Added dropout to reduce the train/val gap
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

class LargeMLP(nn.Module):
    """
    A very large Multi-Layer Perceptron (MLP) architecture.
    Without regularization, this will easily overfit the MNIST training data
    due to its massive capacity (many parameters).
    Notice we DO NOT use Dropout layers here, to adhere to the requirement 
    of not changing the architecture later.
    """
    def __init__(self):
        super(LargeMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------------------
# 3. TRAINING LOOP
# -------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device='cpu'):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
              
    return history

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"\n=> Test Accuracy: {correct / total:.4f}")

def plot_history(history, title):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

# -------------------------------------------------------------------------
# 4. MAIN EXECUTION
# -------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 15 # Train for enough epochs to see overfitting
    criterion = nn.CrossEntropyLoss()

    # =========================================================================
    # PHASE 0: NORMAL MODEL (Baseline)
    # =========================================================================
    print("\n--- PHASE 0: Normal Model Baseline ---")
    print("Using a standard, smaller MLP to show normal behavior.")
    train_loader_norm, val_loader_norm, test_loader_norm = get_dataloaders(use_augmentation=False)
    
    model_norm = NormalMLP().to(device)
    optimizer_norm = optim.Adam(model_norm.parameters(), lr=1e-3)
    
    history_norm = train_model(model_norm, train_loader_norm, val_loader_norm, 
                               optimizer_norm, criterion, epochs=epochs, device=device)
    
    print("\nEvaluating Normal Model on Test Set:")
    evaluate_model(model_norm, test_loader_norm, device)
    plot_history(history_norm, "Phase 0 - Normal Baseline")

    # =========================================================================
    # PHASE 1: FORCE OVERFITTING
    # =========================================================================
    print("\n--- PHASE 1: Training to Overfit ---")
    print("Using large MLP with no regularization and no data augmentation.")
    train_loader_overfit, val_loader_overfit, test_loader_overfit = get_dataloaders(use_augmentation=False)
    
    model_overfit = LargeMLP().to(device)
    # Using Adam without weight decay
    optimizer_overfit = optim.Adam(model_overfit.parameters(), lr=1e-3)
    
    history_overfit = train_model(model_overfit, train_loader_overfit, val_loader_overfit, 
                                  optimizer_overfit, criterion, epochs=epochs, device=device)
    
    print("\nEvaluating Overfitted Model on Test Set:")
    evaluate_model(model_overfit, test_loader_overfit, device)
    plot_history(history_overfit, "Phase 1 - Overfitting")

    # =========================================================================
    # PHASE 2: OVERCOME OVERFITTING (Without changing architecture)
    # =========================================================================
    print("\n--- PHASE 2: Overcoming Overfitting ---")
    print("Using the EXACT SAME LargeMLP architecture.")
    print("Applying L2 Regularization (Weight Decay) and Data Augmentation.")
    
    # 1. Use Data Augmentation
    train_loader_reg, val_loader_reg, test_loader_reg = get_dataloaders(use_augmentation=True)
    
    # 2. Exact same architecture
    model_reg = LargeMLP().to(device)
    
    # 3. Use Weight Decay (L2 regularization) in the optimizer
    # This penalizes large weights, simplifying the model inherently without changing layers.
    optimizer_reg = optim.Adam(model_reg.parameters(), lr=1e-3, weight_decay=1e-3)
    
    history_reg = train_model(model_reg, train_loader_reg, val_loader_reg, 
                              optimizer_reg, criterion, epochs=epochs, device=device)
    
    print("\nEvaluating Regularized Model on Test Set:")
    evaluate_model(model_reg, test_loader_reg, device)
    plot_history(history_reg, "Phase 2 - Regularized")

if __name__ == '__main__':
    main()
