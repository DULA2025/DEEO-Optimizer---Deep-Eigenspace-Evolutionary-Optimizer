import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
# DEEO Optimizer (with E8 Cartan matrix eigendecomposition from DULA framework, enhanced with mod 6 congruence in eigenvalue selection)
class DEEO(optim.Optimizer):
    def __init__(self, params, lr=0.001, iterations=3, beta=0.9):
        defaults = dict(lr=lr, iterations=iterations, beta=beta)
        super(DEEO, self).__init__(params, defaults)
        # E8 Cartan matrix
        cartan = torch.tensor([
            [2, -1, 0, 0, 0, 0, 0, 0],
            [-1, 2, -1, 0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0, 0, 0, 0],
            [0, 0, -1, 2, -1, 0, 0, 0],
            [0, 0, 0, -1, 2, -1, 0, -1],
            [0, 0, 0, 0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0, -1, 2, 0],
            [0, 0, 0, 0, -1, 0, 0, 2]
        ], dtype=torch.float32)
        # Compute eigenvalues and eigenvectors
        eigenvals_complex, eigenvecs_complex = torch.linalg.eig(cartan)
        self.eigenvals = eigenvals_complex.real
        self.eigenvecs = eigenvecs_complex.real
        self.prev_loss = None
        # Device from first param after super
        self.device = torch.device('cpu')
        if self.param_groups and self.param_groups[0]['params']:
            self.device = self.param_groups[0]['params'][0].device
        self.eigenvals = self.eigenvals.to(self.device)
        self.eigenvecs = self.eigenvecs.to(self.device)
        # Initialize momentum buffers if beta > 0
        if beta > 0:
            for group in self.param_groups:
                for p in group['params']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data.view(-1))
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        loss_val = loss.item() if loss is not None else 0
        if self.prev_loss is not None and loss_val > self.prev_loss * 1.1:
            for pg in self.param_groups:
                pg['lr'] *= 0.5
        self.prev_loss = loss_val
        # Vectorize: Collect all grads and params
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
        lr = self.param_groups[0]['lr'] # Assume single group
        iterations = self.param_groups[0]['iterations']
        beta = self.param_groups[0]['beta']
        grad_norm = torch.norm(grad_flat)
        lr_adapt = min(lr, 0.001 / (grad_norm + 1e-5))
        # Chunk grad_flat into segments of 8 (pad last if needed)
        dim = len(grad_flat)
        pad_size = (8 - dim % 8) % 8
        grad_padded = torch.cat([grad_flat, torch.zeros(pad_size, device=self.device)])
        grad_chunks = grad_padded.view(-1, 8) # (num_segments x 8)
        # Project to eigenspace
        grad_eig = torch.matmul(grad_chunks, self.eigenvecs.T)
        # Precondition (tile eigenvals per chunk, with mod 6 enhancement)
        eigenvals_tiled = self.eigenvals.unsqueeze(0).expand(grad_eig.size(0), -1)
        # Enhance stability: boost preconditioner for indices ≡1 or 5 mod 6
        mod6_mask = torch.tensor([i % 6 in [1, 5] for i in range(8)], device=self.device)
        preconditioner = 1 / torch.sqrt(eigenvals_tiled + 1e-3)
        preconditioner[:, mod6_mask] *= 1.1  # Slight boost for "good" congruence classes
        preconditioner = torch.clamp(preconditioner, 0.1, 10)
        update_eig = lr_adapt * (grad_eig * preconditioner)
        # Iterative damping
        for _ in range(iterations):
            update_eig *= 0.9
        # Project back
        update_chunks = torch.matmul(update_eig, self.eigenvecs)
        update_padded = update_chunks.view(-1)
        update = update_padded[:dim] # Unpad
        # Apply momentum if enabled
        if beta > 0:
            all_buffers = [self.state[p]['momentum_buffer'] for p in params_list]
            buffer_flat = torch.cat(all_buffers)
            buffer_flat.mul_(beta).add_(update, alpha=1 - beta)
            update = buffer_flat.clone()
            buffer_idx = 0
            for i, numel in enumerate(numels):
                all_buffers[i].copy_(buffer_flat[buffer_idx:buffer_idx + numel])
                buffer_idx += numel
        # Scatter back
        idx = 0
        for p, numel in zip(params_list, numels):
            with torch.no_grad():
                update_view = update[idx:idx+numel].view_as(p.data)
                p.data.sub_(update_view)
            idx += numel
        return loss
# MLP for EMNIST digits (using matrix multiplications via linear layers)
class EMNISTMLP(nn.Module):
    def __init__(self):
        super(EMNISTMLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
# Data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.EMNIST('.', split='digits', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST('.', split='digits', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
# Model, optimizer, device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EMNISTMLP().to(device)
optimizer = DEEO(list(model.parameters()), lr=0.001, iterations=3, beta=0.9)
# Train function
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
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
# Run training for 500 epochs
for epoch in range(1, 501):
    train(epoch)
test()
