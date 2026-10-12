"""Synthetic counter-access probe; not a model or performance benchmark."""
import torch
assert torch.cuda.get_device_capability()==(12,0)
x=torch.ones(1024,device='cuda',dtype=torch.float32)
y=x+2
torch.cuda.synchronize()
assert bool((y==3).all())
print('SYNTHETIC_COUNTER_TARGET_PASS',flush=True)
