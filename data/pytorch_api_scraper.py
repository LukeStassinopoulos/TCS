import torch
import importlib
import sys

# List of prefixes to exclude
excluded_prefixes = [
    'torch.amp',
    'torch.autograd',
    'torch.library',
    'torch.cpu',
    'torch.cuda',
    'torch.mps',
    'torch.xpu',
    'torch.mtia',
    'torch.backends',
    'torch.export',
    'torch.distributed',
    'torch.distributed.algorithms.join',
    'torch.distributed.elastic',
    'torch.distributed.fsdp',
    'torch.distributed.optim',
    'torch.distributed.pipelining',
    'torch.distributed.tensor.parallel',
    'torch.distributed.checkpoint',
    'torch.compiler',
    'torch.func',
    'torch.futures',
    'torch.fx',
    'torch.fx.experimental',
    'torch.hub',
    'torch.monitor',
    'torch.overrides',
    'torch.package',
    'torch.profiler',
    'torch.nn.attention',
    'torch.onnx',
    'torch.masked',
    'torch.nested',
    'torch.Size',
    'torch.Storage',
    'torch.testing',
    'torch.utils',
    'torch.utils.benchmark',
    'torch.utils.bottleneck',
    'torch.utils.checkpoint',
    'torch.utils.cpp_extension',
    'torch.utils.deterministic',
    'torch.utils.jit',
    'torch.utils.dlpack',
    'torch.utils.mobile_optimizer',
    'torch.utils.model_zoo',
    'torch.utils.tensorboard',
    'torch.utils.module_tracker',
    'torch.ao',
    'torch.quantization',
    'torch.quantize'
]

def traversedir(module, pref='', level=0):
    for attr in dir(module):
        if attr[0] == '_':
            continue
        if attr not in sys.modules:
            fullname = f'{pref}.{attr}'
            if any(fullname.startswith(excluded) for excluded in excluded_prefixes):
                continue 
            yield fullname
            try:
                importlib.import_module(fullname)
                for submodule in traversedir(getattr(module, attr), pref=fullname, level=level+1):
                    yield submodule
            except Exception as e:
                continue

def main():
    with open('api_signatures.txt', 'w') as file:
        for x in traversedir(torch, pref='torch'):
            file.write(f'{x}\n')

if __name__ == "__main__":
    main()
