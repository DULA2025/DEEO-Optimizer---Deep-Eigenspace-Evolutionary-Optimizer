import sympy as sp
import torch
import torch.optim as optim
import numpy as np

# E8 Cartan matrix and eigenvalues (from SymPy)
C = sp.Matrix([
    [2, -1, 0, 0, 0, 0, 0, 0],
    [-1, 2, -1, 0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0, 0, 0, 0],
    [0, 0, -1, 2, -1, 0, 0, 0],
    [0, 0, 0, -1, 2, -1, 0, -1],
    [0, 0, 0, 0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0, -1, 2, 0],
    [0, 0, 0, 0, -1, 0, 0, 2]
])

# Compute numerical eigenvalues and eigenvectors
P, D = C.diagonalize()
eigenvals = torch.tensor([float(d) for d in D.diagonal()])
eigenvecs = torch.tensor(np.array(P).astype(float))  # Approx for basis

# DULA Grading (mod 6 version; extend to mod 10 as needed)
def is_prime(num):
    if num <= 1: return False
    if num <= 3: return True
    if num % 2 == 0 or num % 3 == 0: return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0: return False
        i += 6
    return True

def dula_grading(n: int) -> int:
    m = 0
    factor = 5
    while factor * factor <= n:
        if is_prime(factor) and n % factor == 0:
            count = 0
            while n % factor == 0:
                n //= factor
                count += 1
            if factor % 6 == 5:
                m += count
        factor += 1
    if n > 1 and n % 6 == 5 and is_prime(n):
        m += 1
    return m % 2

# DEEO Optimizer
class DEEO(optim.Optimizer):
    def __init__(self, params, lr=0.01, iterations=10, eigenvals=eigenvals, eigenvecs=eigenvecs):
        defaults = dict(lr=lr, iterations=iterations)
        super(DEEO, self).__init__(params, defaults)
        self.eigenvals = eigenvals
        self.eigenvecs = eigenvecs
        self.device = params[0].device if params else torch.device('cpu')

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            iterations = group['iterations']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                # Adaptive lr (stability trick: like vorticity-adaptive dt)
                grad_norm = torch.norm(grad)
                lr_adapt = min(lr, 0.05 / (grad_norm + 1e-5))
                
                # Project to E8 eigenbasis
                grad_eig = torch.matmul(self.eigenvecs.T.to(self.device), grad.view(-1))
                
                # Precondition with inverse eigenvalues (integer-friendly: avoid sqrt)
                preconditioner = 1 / (self.eigenvals.to(self.device) + 1e-6)
                update_eig = lr_adapt * (grad_eig * preconditioner)
                
                # Iterative projection (stability: multiple inner loops like NS projection)
                for _ in range(iterations):
                    # Mock manifold retraction (e.g., for Stiefel: orthogonalize)
                    update_eig = update_eig - 0.1 * (update_eig.mean() - grad_eig.mean())  # Simple damping
                    
                    # DULA constraint: Grade param and adjust if mismatch (e.g., target even parity)
                    param_int = int(p.data.mean().item())  # Mock int rep
                    grading = dula_grading(abs(param_int))
                    if grading != 0:  # Enforce even m (≡1 mod6)
                        update_eig *= -1  # Flip sign or adjust

                # Back to original basis
                update = torch.matmul(self.eigenvecs.to(self.device), update_eig)
                update = update.view_as(p.data)
                
                # Stable update (no in-place on grad var)
                p.data -= update

        return loss

# Example Usage: Optimize quadratic loss with DEEO
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
param = torch.tensor(np.random.randn(8), device=device, requires_grad=True)  # 8D for E8
optimizer = DEEO([param])

def loss_fn():
    return (param ** 2).sum()

for epoch in range(100):
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}, Param: {param.data}")

# Stability check: Run long, no blow-up
