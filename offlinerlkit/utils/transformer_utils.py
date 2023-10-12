import math
import numpy as np
import torch
import torch.nn as nn

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float

def to_np(x):
	return x.detach().cpu().numpy()


def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	return torch.as_tensor(x, dtype=dtype, device=device)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


class Discretizer:

	def __init__(self, obs_min, obs_max, N):
		self.N = N
		self.min = obs_min
		self.max = obs_max
		self.range = self.max - self.min
		self.step_sizes = self.range / (N - 1)

	def discretize(self, x):
		indices = (x - self.min) / self.step_sizes
		indices = indices.astype(int)
		out_of_bounds = (indices < 0).any(axis=-1) + (indices >= self.N).any(axis=-1)
		if (x[out_of_bounds] != 0).any():
			print('DISCRETIZATION OUT OF BOUNDS')
			indices = np.clip(indices, 0, self.N-1)
		return indices

	def reconstruct(self, indices):
		recon = (indices + .5) * self.step_sizes + self.min
		return recon

	def discretize_torch(self, x):
		x_np = to_np(x)
		return to_torch(self.discretize(x_np), dtype=torch.long)

	def reconstruct_torch(self, indices):
		indices_np = to_np(indices)
		return to_torch(self.reconstruct(indices_np))


	def __call__(self, x):
		if torch.is_tensor(x):
			x_np = to_np(x)
			return_torch = True
		else:
			x_np = x
			return_torch = False
		indices = self.discretize(x_np)
		recon = self.reconstruct(indices)
		error = np.abs(recon - x_np).max(0)
		if return_torch:
			return to_torch(indices, dtype=torch.long, device=x.device), to_torch(recon, device=x.device), to_torch(error, device=x.device)
		else:
			return indices, recon, error


class EinLinear(nn.Module):

    def __init__(self, n_models, in_features, out_features, bias):
        super().__init__()
        self.n_models = n_models
        self.out_features = out_features
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.n_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input):
        """ input : [ B x n_models x input_dim ] """
        # [ B x n_models x output_dim ]
        output = torch.einsum('eoi,bei->beo', self.weight, input)
        if self.bias is not None:
            raise RuntimeError()
        return output

    def extra_repr(self):
        return 'n_models={}, in_features={}, out_features={}, bias={}'.format(
            self.n_models, self.in_features, self.out_features, self.bias is not None
        )