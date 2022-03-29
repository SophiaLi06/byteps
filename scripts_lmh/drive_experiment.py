import torch
import numpy as np
import math

def new_hadamard(arr):
  print("new hadamard")
  length = len(arr)
  res = [0.0] * length
  for i in range(length):
    res[i] = arr[i]
  h = 2
  while h <= length:
    hf = h // 2
    for i in range(length // h):
      for j in range(hf):
        res[i * h + j] = res[i * h + j] + res[i * h + hf + j]
        res[i * h + hf + j] = res[i * h + j] - 2 * res[i * h + hf + j]
    h *= 2

  sqrt_d = math.sqrt(length)
  for i in range(length):
    res[i] /= sqrt_d

  norm1 = 0
  norm2 = 0
  for i in range(length):
    norm1 += abs(res[i])
    norm2 += (res[i] * res[i])
  scale = norm2 / norm1
  return res, scale



##############################################################################
##############################################################################

def hadamard_rotate(vec):
  '''
  In-place 1D hadamard transform 
  '''
    
  numel = vec.numel()
  print(numel)
  if numel & (numel-1) != 0:
      raise Exception("vec numel must be a power of 2")
      
  h = 2

  while h <= numel:
      
    hf = h // 2
    vec = vec.view(numel // h, h)

    vec[:, :hf] = vec[:, :hf] + vec[:, hf:2 * hf]
    vec[:, hf:2 * hf] = vec[:, :hf] - 2 * vec[:, hf:2 * hf]
    h *= 2

  vec /= np.sqrt(numel)

##############################################################################
##############################################################################

def drive_compress(vec, prng=None):
  '''
  :param vec: the vector to compress (currently we require vec numel to be a power of two)
  :param prng: a generator that determines the specific (random) Hadamard rotation
  :return: compressed vector
  '''
  
  ### dimension
  numel = vec.numel()
  
  ### in-place hadamard transform
  ######Minghao
  #rand_arr = torch.bernoulli(torch.ones(numel, device=vec.device) / 2, generator=prng)
  #print(rand_arr)
  if prng is not None:
    #vec = vec * (2 * rand_arr - 1)
    vec = vec * (2 * torch.bernoulli(torch.ones(numel, device=vec.device) / 2, generator=prng) - 1)
  hadamard_rotate(vec)
  ##### Minghao
  print(vec)

  #### compute the scale (rotation preserves the L2 norm)
  scale = torch.norm(vec, 2) ** 2 / torch.norm(vec, 1)

  ##### take the sign
  vec = 1.0 - 2 * (vec < 0)

  #### send
  return vec, scale

##############################################################################

def drive_decompress(vec, scale, prng=None):
  '''
    :param assignments: sign(Rx) from the paper
    :param scale: S from the paper
    :param prng: random generator for Hadamard rotation, should have the same state used for compression
    :return: decompressed vector
    '''

  ### dimension
  numel = vec.numel()

  ### in-place hadamard transform (inverse)
  hadamard_rotate(vec)
  if prng is not None:
    vec = vec * (2 * torch.bernoulli(torch.ones(numel, device=vec.device) / 2, generator=prng) - 1)

  ##### scale and return
  return scale * vec

##############################################################################
##############################################################################

if __name__ == "__main__":
  a = torch.rand(512, 512)
  print(a)
  #b = a.numpy().flatten()
  #print(b)
  #seed = np.random.randint(2 ** 31)
  #rgen = torch.Generator(device='cpu')
  #rgen.manual_seed(seed)
  #vec, scale = drive_compress(a, rgen)
  vec, scale = drive_compress(a)
  print(vec, scale)
  #print(drive_decompress(vec, scale))
  #print(b)
  #print(new_hadamard(b))