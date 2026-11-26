"""Explicit vector backends; CUDA failures never select a different device."""
import numpy as np


def unit_rows(value):
    matrix = np.atleast_2d(np.asarray(value, dtype=np.float64))
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError('Expected a finite vector matrix')
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('Zero vector')
    return matrix / norms


def stable_topk(scores, ids, k):
    values = np.asarray(scores)
    if (type(k) is not int or k < 1 or values.ndim != 1 or len(values) != len(ids)
            or not np.isfinite(values).all() or len(set(ids)) != len(ids)):
        raise ValueError('Invalid ranking inputs')
    return sorted(range(len(ids)), key=lambda i: (-float(values[i]), ids[i]))[:k]


class NumpyBackend:
    def __init__(self, dtype='float64'):
        if dtype not in ('float32', 'float64'):
            raise ValueError('Unsupported score dtype')
        self.dtype = np.dtype(dtype)

    def prepare(self, value):
        matrix = np.atleast_2d(np.asarray(value, dtype=self.dtype))
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError('Expected a finite vector matrix')
        return matrix

    def take(self, matrix, indices):
        return matrix[indices]

    def scores(self, queries, candidates):
        return self.prepare(queries) @ self.prepare(candidates).T

    def synchronize(self):
        pass

    def info(self):
        return {'backend': 'numpy', 'device': 'cpu', 'score_dtype': self.dtype.name}


class CUDABackend(NumpyBackend):
    def __init__(self, device='cuda:0', dtype='float64'):
        super().__init__(dtype)
        import torch
        if not device.startswith('cuda:') or not torch.cuda.is_available():
            raise RuntimeError('CUDA backend required; CPU fallback is disabled')
        self.torch, self.device = torch, device
        self.torch_dtype = torch.float64 if dtype == 'float64' else torch.float32
        # Resolve/allocate now so invalid devices fail before a run is published.
        torch.empty(0, device=device)

    def prepare(self, value):
        if self.torch.is_tensor(value):
            if not bool(self.torch.isfinite(value).all()):
                raise ValueError('Nonfinite CUDA vectors')
            return value.to(device=self.device, dtype=self.torch_dtype)
        return self.torch.as_tensor(super().prepare(value), device=self.device, dtype=self.torch_dtype)

    def scores(self, queries, candidates):
        with self.torch.inference_mode():
            result = self.prepare(queries) @ self.prepare(candidates).T
        return result.cpu().numpy()

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def info(self):
        return {'backend': 'torch_cuda', 'device': self.device, 'score_dtype': self.dtype.name,
                'torch': self.torch.__version__, 'cuda_build': self.torch.version.cuda,
                'gpu': self.torch.cuda.get_device_name(self.device)}


def make_vector_backend(config):
    name = config.get('backend', 'torch_cuda')
    dtype = config.get('score_dtype', 'float64')
    if name == 'numpy':
        return NumpyBackend(dtype)
    if name == 'torch_cuda':
        return CUDABackend(config.get('device', 'cuda:0'), dtype)
    raise ValueError('Unknown vector backend')
