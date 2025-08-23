import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# DEEO Optimizer (Fixed Vectorized version with optional momentum)
class DEEO(optim.Optimizer):
    def __init__(self, params, lr=0.001, iterations=3, beta=0.0):
        defaults = dict(lr=lr, iterations=iterations, beta=beta)
        super(DEEO, self).__init__(params, defaults)
        self.eigenvals = torch.tensor([0.01, 0.5, 1.2, 1.6, 2.4, 2.8, 3.5, 4.0], dtype=torch.float32)
        self.eigenvecs = torch.eye(8, dtype=torch.float32)
        self.prev_loss = None

        # Fixed device detection
        self.device = torch.device('cpu')
        for group in self.param_groups:
            for p in group['params']:
                if p is not None:
                    self.device = p.device
                    break
            if self.device != torch.device('cpu'):
                break

        self.eigenvals = self.eigenvals.to(self.device)
        self.eigenvecs = self.eigenvecs.to(self.device)
        
        # Initialize momentum buffers if beta > 0
        if self.param_groups[0]['beta'] > 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p is not None:
                        self.state[p]['momentum_buffer'] = torch.zeros_like(p.data.view(-1))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Adaptive learning rate based on loss trend
        loss_val = loss.item() if loss is not None else 0
        if self.prev_loss is not None and loss_val > self.prev_loss * 1.1:
            for pg in self.param_groups:
                pg['lr'] *= 0.5
        self.prev_loss = loss_val

        # Collect all gradients and parameters
        grads = []
        params_list = []
        shapes = []
        numels = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads.append(p.grad.data.view(-1))
                params_list.append(p)
                shapes.append(p.shape)
                numels.append(p.numel())

        if not grads:
            return loss

        grad_flat = torch.cat(grads)
        lr = self.param_groups[0]['lr']
        iterations = self.param_groups[0]['iterations']
        beta = self.param_groups[0]['beta']

        # Gradient norm clipping for stability
        grad_norm = torch.norm(grad_flat)
        lr_adapt = min(lr, 0.001 / (grad_norm + 1e-8))

        # Chunk gradient into segments of 8 (pad if necessary)
        dim = len(grad_flat)
        pad_size = (8 - dim % 8) % 8
        if pad_size > 0:
            grad_padded = torch.cat([grad_flat, torch.zeros(pad_size, device=self.device)])
        else:
            grad_padded = grad_flat
        grad_chunks = grad_padded.view(-1, 8)  # (num_segments, 8)

        # Fixed matrix multiplication: Project to eigenspace
        # grad_chunks: (N, 8), eigenvecs.T: (8, 8) -> result: (N, 8)
        grad_eig = torch.matmul(grad_chunks, self.eigenvecs.T)

        # Apply preconditioning using eigenvalues
        eigenvals_expanded = self.eigenvals.unsqueeze(0).expand_as(grad_eig)
        preconditioner = 1.0 / torch.sqrt(eigenvals_expanded + 1e-8)
        preconditioner = torch.clamp(preconditioner, 0.1, 10.0)
        update_eig = lr_adapt * (grad_eig * preconditioner)

        # Iterative damping for stability
        for _ in range(iterations):
            update_eig *= 0.9

        # Project back to parameter space
        # update_eig: (N, 8), eigenvecs: (8, 8) -> result: (N, 8)
        update_chunks = torch.matmul(update_eig, self.eigenvecs)
        update_padded = update_chunks.view(-1)
        update = update_padded[:dim]  # Remove padding

        # Apply momentum if enabled
        if beta > 0:
            # Collect momentum buffers
            all_buffers = []
            for p in params_list:
                buffer = self.state[p]['momentum_buffer']
                all_buffers.append(buffer)
            
            if all_buffers:
                buffer_flat = torch.cat(all_buffers)
                # Update momentum
                buffer_flat.mul_(beta).add_(update, alpha=1 - beta)
                update = buffer_flat.clone()
                
                # Update individual buffers
                buffer_idx = 0
                for i, p in enumerate(params_list):
                    numel = numels[i]
                    self.state[p]['momentum_buffer'] = buffer_flat[buffer_idx:buffer_idx + numel]
                    buffer_idx += numel

        # Update parameters
        param_idx = 0
        for p, numel in zip(params_list, numels):
            with torch.no_grad():
                update_view = update[param_idx:param_idx + numel].view_as(p.data)
                p.data.sub_(update_view)
            param_idx += numel

        return loss

# Improved CNN for MNIST with better architecture
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Input: 28x28
        x = F.relu(self.conv1(x))  # 28x28
        x = F.max_pool2d(x, 2)     # 14x14
        x = F.relu(self.conv2(x))  # 14x14
        x = F.max_pool2d(x, 2)     # 7x7
        x = F.relu(self.conv3(x))  # 7x7
        
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Data loaders with improved transforms
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST('.', train=False, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

# Model, optimizer, device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MNISTCNN().to(device)
# Try with and without momentum
optimizer = DEEO(list(model.parameters()), lr=0.01, iterations=3, beta=0.9)

# Training function with better logging
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    train_loss /= len(train_dataset)
    acc = 100. * correct / len(train_dataset)
    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Accuracy: {acc:.2f}%')
    return train_loss, acc

# Test function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_dataset)
    acc = 100. * correct / len(test_dataset)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {acc:.2f}%')
    return test_loss, acc

# Main training loop - reasonable number of epochs for MNIST
if __name__ == "__main__":
    print("Starting MNIST training with DEEO optimizer...")
    
    num_epochs = 200  # MNIST typically converges quickly
    best_test_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test()
        
        # Track best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"New best test accuracy: {best_test_acc:.2f}%")
        
        print(f"Epoch {epoch}/{num_epochs} - Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        print("-" * 60)
        
        # Early stopping if we reach good accuracy
        if test_acc >= 99.0:
            print(f"Reached {test_acc:.2f}% accuracy, stopping early!")
            break
    
    print(f"Training completed! Best test accuracy: {best_test_acc:.2f}%")
