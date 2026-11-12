#!/usr/bin/env python3
"""Run inside each target model environment and save the output with its logs."""
import json
import platform
import torch

if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable; check GPU access, driver and PyTorch build')
result = dict(python=platform.python_version(), torch=torch.__version__,
              torch_cuda=torch.version.cuda, compiled_arches=torch.cuda.get_arch_list(), gpus=[])
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        x = torch.randn(256, 256, device='cuda', requires_grad=True)
        loss = (x @ x).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        result['gpus'].append(dict(index=index, name=torch.cuda.get_device_name(index),
                                   capability=torch.cuda.get_device_capability(index),
                                   memory_gib=round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2),
                                   forward_backward='passed'))
print(json.dumps(result, indent=2))
