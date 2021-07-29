import torch
from torch.autograd import Variable

class Model(torch.nn.Module):
	def __init__(
		self,
		num_vertices,
		timesteps,
		emb_dim,
		params={},
		device=torch.device('cpu')
	):
		super(Model, self).__init__()

		self.num_vertices = num_vertices
		self.timesteps = timesteps
		self.emb_dim = emb_dim

		self.embedding = Variable(
			torch.randn(num_vertices, timesteps, emb_dim),
			requires_grad=True,
			device=device
		)
		self.theta = Variable(
			torch.randn(emb_dim),
			requires_grad=True,
			device=device
		)
		self.beta = = Variable(
			torch.randn(1),
			requires_grad=True,
			device=device
		)

		self.params['beta_triad'] = self.params.get('beta_triad', 1)
		self.params['beta_smooth'] = self.params.get('beta_smooth', 1)

	def forward(self, data, weight, triag_int, triag_float):
		"""
		:param data: (batchsize, 5), [k, from_pos, to_pos, from_neg, to_neg]
		:param weight: (batchsize, )
		:param triag_int: (batchsize, 4), [k, from, to1, to2]
		:param triag_float: (batchsize, 3), [coef, w1, w2]
		"""

		# (batchsize, d) => (batchsize, )
		dist_pos = embedding[data[:, 0], data[:, 1]] - embedding[data[:, 0], data[:, 2]]
		dist_pos = torch.sum(dist_pos * dist_pos, axis=-1)
		dist_neg = embedding[data[:, 0], data[:, 3]] - embedding[data[:, 0], data[:, 4]]
		dist_neg = torch.sum(dist_neg * dist_neg, axis=-1)

		lprox = torch.maximum(dist_pos - dist_neg + 1, 0) * weight
		lprox = K.mean(lprox)

		lsmooth = embedding[1:] - embedding[:-1]  # (k - 1, nsize, d)
		lsmooth = torch.sum(torch.square(lsmooth), axis=-1)  # (k - 1, nsize)
		lsmooth = torch.mean(lsmooth)

		e1 = embedding[triag_int[:, 0], triag_int[:, 1]] - embedding[triag_int[:, 0], triag_int[:, 2]]  # (batchsize_t, d)
		e2 = embedding[triag_int[:, 0], triag_int[:, 1]] - embedding[triag_int[:, 0], triag_int[:, 3]]
		x = e1 * triag_float[:, 1, None] + e2 * triag_float[:, 2, None]

		iprod = x.dot(torch.expand_dims(self.theta, axis=1)) + self.beta # (batchsize_d, )
		iprod = torch.clip(iprod, -50, 50)  # for numerical stability
		logprob = torch.log(1 + K.exp(-iprod))

		ltriag = K.mean(triag_float[:, 0] * iprod + logprob)

		loss = lprox + self.params['beta_smooth'] * lsmooth + self.params['beta_triad'] * ltriag

		return loss

	def predict(self, pred_data):
		"""
		:param pred_data: (batchsize, 2)  [timestep, nodeid]
		"""

		# (batchsize, nsize, d) => (batchsize, nsize)
		pred = self.embedding[pred_data[:, 0] - 1, pred_data[:, 1]][:, None, :] - self.embedding[pred_data[:, 0] - 1]
		pred = -torch.sum(torch.square(pred), axis=-1)  # the closer the more probable

		return pred
