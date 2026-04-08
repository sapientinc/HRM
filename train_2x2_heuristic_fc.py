import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import wandb
from tqdm import tqdm

class Config:
    # Data
    data_path = "data/cube-2-by-2-heuristic"
    max_train_samples = 1000
    max_val_samples = 100
    max_test_samples = None
    
    # Model architecture
    input_size = 24 * 6  # 24 positions, one-hot encoded with 6 colors
    hidden_size = 512
    num_layers = 4
    output_size = 1
    
    # Training
    batch_size = 256
    epochs = 20
    learning_rate = 1e-3
    weight_decay = 1e-4
    
    # Logging and checkpoints
    project_name = "2x2-cube-heuristic-fc"
    checkpoint_dir = "checkpoints/2x2-heuristic-fc"
    checkpoint_every = 5 # epochs
    log_freq = 100  # Log every N steps
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random seed
    seed = 42


class CubeHeuristicDataset(Dataset):    
    def __init__(self, data_path: str, split: str = "train", max_samples: int = None):
        self.data_path = Path(data_path) / split
        
        # Load data
        self.inputs = np.load(self.data_path / "all__inputs.npy")
        self.labels = np.load(self.data_path / "all__labels.npy")
        
        # Subsample
        total_samples = len(self.inputs)
        if max_samples is not None and max_samples < total_samples:
            # Random samples
            rng = np.random.RandomState(42)
            indices = rng.choice(total_samples, size=max_samples, replace=False)
            self.inputs = self.inputs[indices]
            self.labels = self.labels[indices]
            print(f"Subsampled to {max_samples} examples")
        
        # Load metadata
        with open(self.data_path / "dataset.json", "r") as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {split} set: {len(self.inputs)} examples")
        print(f"  Input shape: {self.inputs.shape}")
        print(f"  Label shape: {self.labels.shape}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # Get cube state
        state = self.inputs[idx]
        
        # One-hot encode: 24 positions x 6 colors = 144 features
        one_hot = np.zeros((24, 6), dtype=np.float32)
        for i, color in enumerate(state):
            one_hot[i, color] = 1.0
        one_hot = one_hot.flatten()
        
        # Get label
        label = self.labels[idx].astype(np.float32)
        label = label[0]
        
        return torch.from_numpy(one_hot), torch.tensor(label, dtype=torch.float32)


class HeuristicNet(nn.Module):    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


def save_checkpoint(model, optimizer, epoch, step, train_loss, val_loss, path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['step']


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, config):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    global_step = epoch * len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            mae = torch.abs(outputs - labels).mean()
        
        total_loss += loss.item()
        total_mae += mae.item()
        num_batches += 1
        global_step += 1
        
        # Log to wandb
        wandb.log({
            'train/batch_loss': loss.item(),
            'train/batch_mae': mae.item(),
            'train/step': global_step,
        }, step=global_step)
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    return avg_loss, avg_mae, global_step


def validate(model, val_loader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_correct = 0  # Prediction rounded to nearest int match label
    num_batches = 0
    num_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            mae = torch.abs(outputs - labels).mean()
            
            # Compute accuracy
            preds_rounded = torch.round(outputs)
            correct = (preds_rounded == labels).sum().item()
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_correct += correct
            num_batches += 1
            num_samples += labels.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    accuracy = total_correct / num_samples
    
    # Compute additional metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    rmse = np.sqrt(((all_preds - all_labels) ** 2).mean())
    
    return {
        'loss': avg_loss,
        'mae': avg_mae,
        'rmse': rmse,
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
    }


def main():
    config = Config()
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_checkpoint_dir = Path(config.checkpoint_dir) / timestamp
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project=config.project_name,
        config={
            'input_size': config.input_size,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'output_size': config.output_size,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'seed': config.seed,
            'max_train_samples': config.max_train_samples,
            'max_val_samples': config.max_val_samples,
            'max_test_samples': config.max_test_samples,
        }
    )
    
    print(f"Wandb run: {run.name}")
    print(f"Device: {config.device}")
    print(f"Checkpoints will be saved to: {run_checkpoint_dir}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CubeHeuristicDataset(config.data_path, split="train", max_samples=config.max_train_samples)
    val_dataset = CubeHeuristicDataset(config.data_path, split="val", max_samples=config.max_val_samples)
    test_dataset = CubeHeuristicDataset(config.data_path, split="test", max_samples=config.max_test_samples)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = HeuristicNet(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size
    )
    model = model.to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    wandb.watch(model, log='all', log_freq=config.log_freq)
    
    # Create optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs, 
        eta_min=config.learning_rate * 0.01
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        
        # Train
        train_loss, train_mae, global_step = train_epoch(
            model, train_loader, optimizer, criterion, 
            config.device, epoch, config
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config.device, config)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log epoch metrics
        wandb.log({
            'epoch': epoch + 1,
            'train/epoch_loss': train_loss,
            'train/epoch_mae': train_mae,
            'val/loss': val_metrics['loss'],
            'val/mae': val_metrics['mae'],
            'val/rmse': val_metrics['rmse'],
            'val/accuracy': val_metrics['accuracy'],
            'learning_rate': current_lr,
        }, step=global_step)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = run_checkpoint_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, epoch, global_step,
                train_loss, val_metrics['loss'], best_path
            )
            wandb.log({'best_val_loss': best_val_loss}, step=global_step)
            print(f"  New best model saved! Val Loss: {best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.checkpoint_every == 0:
            checkpoint_path = run_checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                model, optimizer, epoch, global_step,
                train_loss, val_metrics['loss'], checkpoint_path
            )
    
    # Save final model
    final_path = run_checkpoint_dir / "final_model.pt"
    save_checkpoint(
        model, optimizer, config.epochs - 1, global_step,
        train_loss, val_metrics['loss'], final_path
    )
    
    # Validate using test set
    test_metrics = validate(model, test_loader, criterion, config.device, config)
    
    wandb.log({
        'test/loss': test_metrics['loss'],
        'test/mae': test_metrics['mae'],
        'test/rmse': test_metrics['rmse'],
        'test/accuracy': test_metrics['accuracy'],
    })
    
    wandb.log({
        'test/predictions_histogram': wandb.Histogram(test_metrics['predictions']),
        'test/labels_histogram': wandb.Histogram(test_metrics['labels']),
    })
    
    wandb.finish()
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {run_checkpoint_dir}")


if __name__ == "__main__":
    main()
