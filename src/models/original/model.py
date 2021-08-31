import torch
import numpy as np
from torch.autograd import Variable
from loguru import logger

class Model(torch.nn.Module):
	def __init__(
		self,
		num_vertices,
		timesteps,
		emb_dim,
		params={},
		device='cpu'
	):
		super(Model, self).__init__()

		self.device = device

		self.num_vertices = num_vertices
		self.timesteps = timesteps
		self.emb_dim = emb_dim

		embedding = torch.randn(timesteps, num_vertices, emb_dim, dtype=torch.double)
		theta = torch.randn(emb_dim, dtype=torch.double)
		beta = torch.randn(1, dtype=torch.double)

		if device == 'cuda':
			embedding = embedding.cuda()
			theta = theta.cuda()
			beta = beta.cuda()

		self.embedding = Variable(embedding, requires_grad=True)
		self.theta = Variable(theta, requires_grad=True)
		self.beta = Variable(beta, requires_grad=True)

		self.params = {}
		self.params['beta_triad'] = params.get('beta_triad', 1)
		self.params['beta_smooth'] = params.get('beta_smooth', 1)

	def forward(self, data, weight, triag_int, triag_float):
		"""
		:param data: (batchsize, 5), [k, from_pos, to_pos, from_neg, to_neg]
		:param weight: (batchsize, )
		:param triag_int: (batchsize, 4), [k, from, to1, to2]
		:param triag_float: (batchsize, 3), [coef, w1, w2]
		"""

		# (batchsize, d) => (batchsize, )
		dist_pos = self.embedding[data[:, 0], data[:, 1]] - self.embedding[data[:, 0], data[:, 2]]
		dist_pos = torch.sum(dist_pos * dist_pos, axis=-1)
		dist_neg = self.embedding[data[:, 0], data[:, 3]] - self.embedding[data[:, 0], data[:, 4]]
		dist_neg = torch.sum(dist_neg * dist_neg, axis=-1)

		diff = dist_pos - dist_neg + 1
		zero = dist_pos - dist_pos
		maximum = torch.max(diff, zero)
		lprox = maximum * weight
		lprox = torch.mean(lprox)

		lsmooth = self.embedding[1:] - self.embedding[:-1]  # (k - 1, nsize, d)
		lsmooth = torch.sum(torch.square(lsmooth), axis=-1)  # (k - 1, nsize)
		lsmooth = torch.mean(lsmooth)

		e1 = self.embedding[triag_int[:, 0], triag_int[:, 1]] - self.embedding[triag_int[:, 0], triag_int[:, 2]]  # (batchsize_t, d)
		e2 = self.embedding[triag_int[:, 0], triag_int[:, 1]] - self.embedding[triag_int[:, 0], triag_int[:, 3]]
		x = e1 * triag_float[:, 1, None] + e2 * triag_float[:, 2, None]

		repeated = self.theta.repeat(self.theta.shape[0], 1)

		iprod = torch.mm(x, repeated)
		iprod += self.beta # (batchsize_d, )
		iprod = torch.clip(iprod, -50, 50)  # for numerical stability
		logprob = torch.log(1 + torch.exp(-iprod))

		C = triag_float[:, 0]
		C = C.view((C.shape[0], 1))

		ltriag = torch.mean(C * iprod + logprob)

		loss = lprox + self.params['beta_smooth'] * lsmooth + self.params['beta_triad'] * ltriag

		return loss

	def parameters(self, recurse=True):
		return [self.embedding, self.theta, self.beta]
