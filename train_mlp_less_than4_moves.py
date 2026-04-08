"""
Train MLP on short sequences (<=4 moves) to test if unique optimal
solutions make learning easier than longer scrambles with multiple solutions.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


class SimpleMLP(nn.Module):
    """4-layer MLP with one-hot encoding for sequence prediction."""
    def __init__(self, input_dim=144, hidden_dim=256, seq_len=15, vocab_size=10, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, seq_len * vocab_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def forward(self, x):
        # Convert state indices (1-6) to one-hot (0-5)
        x = (x.long() - 1).clamp(min=0, max=5)
        x_onehot = torch.nn.functional.one_hot(x, num_classes=6).float()
        x_onehot = x_onehot.reshape(x_onehot.shape[0], -1)
        
        x = torch.nn.functional.relu(self.fc1(x_onehot))
        if self.dropout:
            x = self.dropout(x)
        
        x = torch.nn.functional.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout(x)
            
        x = torch.nn.functional.relu(self.fc3(x))
        if self.dropout:
            x = self.dropout(x)
            
        x = self.fc4(x)
        return x.view(x.shape[0], self.seq_len, self.vocab_size)


def load_dataset_split(data_dir: Path, split: str, max_length: int = None):
    """Load dataset split, optionally filtering by sequence length."""
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split {split} not found in {data_dir}")

    with open(split_dir / "dataset.json", "r") as f:
        metadata = json.load(f)

    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy").squeeze()
    
    if max_length is not None:
        seq_lengths = (labels != 0).sum(axis=1)
        mask = seq_lengths <= max_length
        inputs = inputs[mask]
        labels = labels[mask]
        print(f"  Filtered {split}: {mask.sum()}/{len(mask)} samples (<={max_length} moves)")

    return {'metadata': metadata, 'inputs': inputs, 'labels': labels}


def eval_epoch(model, loader, criterion, device):
    """Evaluate model on a loader in eval mode (no dropout)."""
    model.eval()
    total_loss = 0.0
    token_correct = 0
    token_total = 0
    seq_correct = 0
    count = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
            total_loss += loss.item() * inputs.size(0)
            
            preds = torch.argmax(logits, dim=-1)
            mask = targets != 0
            token_correct += (preds.eq(targets) & mask).sum().item()
            token_total += mask.sum().item()
            seq_correct += (preds.eq(targets) | ~mask).all(dim=1).sum().item()
            count += inputs.size(0)
    
    avg_loss = total_loss / count
    token_acc = token_correct / max(1, token_total)
    seq_acc = seq_correct / count
    return avg_loss, token_acc, seq_acc


def train_model(
    model,
    train_data,
    test_data,
    epochs=25,
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=128,
    device='cpu',
):
    """Train MLP with CrossEntropyLoss."""
    def format_time(seconds):
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"
    
    train_dataset = TensorDataset(
        torch.from_numpy(train_data['inputs']),
        torch.from_numpy(train_data['labels'].astype(np.int64))
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_data['inputs']),
        torch.from_numpy(test_data['labels'].astype(np.int64))
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_token_acc': [],
        'test_token_acc': [],
        'train_seq_acc': [],
        'test_seq_acc': []
    }
    
    start_time = time.time()
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        # Train
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()
        
        # Evaluate on train and test sets in eval mode
        train_loss, train_token_acc, train_seq_acc = eval_epoch(model, train_loader, criterion, device)
        test_loss, test_token_acc, test_seq_acc = eval_epoch(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_token_acc'].append(train_token_acc)
        history['test_token_acc'].append(test_token_acc)
        history['train_seq_acc'].append(train_seq_acc)
        history['test_seq_acc'].append(test_seq_acc)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch_time * (epochs - epoch - 1)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
              f"Train Token Acc={train_token_acc:.4f}, Test Token Acc={test_token_acc:.4f}, "
              f"Test Seq Acc={test_seq_acc:.4f}, "
              f"Epoch Time={format_time(epoch_time)}, ETA={format_time(eta)}")
    
    return history, time.time() - start_time


if __name__ == "__main__":
    MAX_SEQUENCE_LENGTH = 4
    
    print("Training MLP on short sequences only")
    print(f"Max sequence length: {MAX_SEQUENCE_LENGTH} moves")
    print("Testing if unique solutions help learning")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    data_dir = Path("data/cube-2-by-2")
    
    print("\nLoading dataset...")
    train_data = load_dataset_split(data_dir, "train", max_length=MAX_SEQUENCE_LENGTH)
    test_data = load_dataset_split(data_dir, "test", max_length=MAX_SEQUENCE_LENGTH)
    print(f"Final counts - Train: {len(train_data['inputs'])}, Test: {len(test_data['inputs'])}")
    
    seq_len = train_data['labels'].shape[1]
    vocab_size = train_data['metadata']['vocab_size']
    print(f"\nCreating MLP (4 layers, hidden=256, seq_len={seq_len}, vocab={vocab_size})")
    model = SimpleMLP(hidden_dim=256, seq_len=seq_len, vocab_size=vocab_size, dropout=0.1)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    print("\nTraining...")
    history, total_time = train_model(
        model, train_data, test_data,
        epochs=25, lr=1e-3, weight_decay=1e-5,
        batch_size=128, device=device
    )
    
    # Save plot
    output_dir = Path(r"C:\personal\Uni\5\ML\Project\HRM\2x2_solution_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"mlp_short_seq_len{MAX_SEQUENCE_LENGTH}_results.png"

    epochs_range = np.arange(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_range, history['train_loss'], label="Train")
    axes[0].plot(epochs_range, history['test_loss'], label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history['train_token_acc'], label="Train")
    axes[1].plot(epochs_range, history['test_token_acc'], label="Test")
    axes[1].set_title("Token Accuracy (non-PAD)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_range, history['train_seq_acc'], label="Train")
    axes[2].plot(epochs_range, history['test_seq_acc'], label="Test")
    axes[2].set_title("Sequence Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"MLP Short Seq (<={MAX_SEQUENCE_LENGTH}) | {num_params:,} params | {total_time:.1f}s | "
        f"lr=1e-3 wd=1e-5 batch=128", fontsize=12
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    
    print("\n" + "-"*50)
    print("Training complete")
    print("-"*50)
    print(f"Test Loss: {history['test_loss'][-1]:.4f}")
    print(f"Test Token Acc: {history['test_token_acc'][-1]:.4f}")
    print(f"Test Seq Acc: {history['test_seq_acc'][-1]:.4f}")
    print(f"Train Token Acc: {history['train_token_acc'][-1]:.4f}")
    print(f"Overfitting Gap: {history['test_loss'][-1] - history['train_loss'][-1]:.4f}")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Parameters: {num_params:,}")
    print(f"\nPlot saved: {output_file}")
