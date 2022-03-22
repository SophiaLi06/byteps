import torch
import time

x = torch.rand(100000)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

gpu_tensor = x.to(device)
start = time.time()
b = torch.bernoulli(gpu_tensor)
end = time.time()
print(end - start)

start1 = time.time()
for i in range(100000):
    torch.bernoulli(gpu_tensor[i])
print(time.time() - start1)
