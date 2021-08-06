import math
import torch
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
from models.original.model import Model

class Trainer:
	def __init__(self, model, dataset):
		self.model = model
		self.dataset = dataset

	def train(self, lr=0.1, epochs=10, batchsize=1000, negdup=1):
		optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		positive_samples, weight = self.gen_positive_samples()

		total_batches = math.ceil(positive_samples.shape[0]/batchsize)
		logger.debug(total_batches)

		pbar = tqdm(range(epochs), position=0, leave=False, desc='epoch')
		for epoch in pbar:
			negative_social_homophily_samples = self.gen_social_homophily_samples(positive_samples)
			triad_data = self.gen_triad_samples(positive_samples)
			emcoef_int, emcoef_float = self.calculate_EM_coefficient(triad_data)

			data = []
			for i in range(len(positive_samples)):
				for j in range(0, len(negative_social_homophily_samples[i]), 2):
					d = list(positive_samples[i])
					d.append(negative_social_homophily_samples[i][j])
					d.append(negative_social_homophily_samples[i][j+1])
					data.append(d)

			data = np.array(data)

			ave_loss = 0

			pbar2 = tqdm(
				self.gen_batches(data, weight, emcoef_int, emcoef_float, batchsize),
				position=1, leave=False, desc='batch'
			)
			for batch in pbar2:
				data_batch, weight_batch, emcoef_int_batch, emcoef_float_batch = batch

				optimizer.zero_grad()
				loss = self.model(
					data_batch,
					weight_batch,
					emcoef_int_batch,
					emcoef_float_batch
				)
				loss.backward()
				optimizer.step()

				ave_loss += loss / total_batches

			pbar.write('Epoch {:02}: {:.4} training loss'.format(epoch, ave_loss.item()))
			pbar.update()

		pbar.close()

		return self.model

	def gen_batches(self, datas, weight, emcoef_int, emcoef_float, batchsize):
		curstart = 0
		while True:
			if curstart >= len(datas):
				break

			data_batch = datas[curstart:min(len(datas), curstart+batchsize)]
			weight_batch = weight[curstart:min(len(weight), curstart+batchsize)]
			emcoef_int_batch = emcoef_int[curstart:min(len(emcoef_int), curstart+batchsize)]
			emcoef_float_batch = emcoef_float[curstart:min(len(emcoef_float), curstart+batchsize)]

			yield data_batch, weight_batch, emcoef_int_batch, emcoef_float_batch
			curstart += batchsize

	def gen_positive_samples(self):
		positive_samples, weight = [], []

		for i, graph in enumerate(self.dataset):
			if i >= self.model.timesteps:
				break
			assert graph.undirected()

			for e in graph.edges():
				source, target = e[0], e[1]
				if source > target:
					source, target = target, source

				positive_samples.append([i, self.dataset.number2idx[source], self.dataset.number2idx[target]])
				weight.append(graph.get_edge_weight(source, target))

		positive_samples = np.array(positive_samples, dtype='int32')
		weight = np.array(weight, dtype='float32')

		assert len(positive_samples) > 0, "No positive sample is generated given an empty graph"
		return positive_samples, weight

	def gen_social_homophily_samples(self, positive_samples, negdup=1):
		negative_social_homophily_samples = []

		for k, source_index, target_index in positive_samples:
			source = self.dataset.vertices[source_index]
			target = self.dataset.vertices[target_index]
			graph = self.dataset[k]
			social_homophily_sample = []

			for _ in range(negdup):
				if random.randrange(2) == 0:  # replace source
					new_source = random.choice(graph.vertices())
					while graph.has_edge(new_source, target):
						new_source = random.choice(graph.vertices())
					new_source_index = self.dataset.number2idx[new_source]
					social_homophily_sample.extend([new_source_index, target_index])

				else:  # replace target
					new_target = random.choice(graph.vertices())
					while graph.has_edge(source, new_target):
						new_target = random.choice(graph.vertices())
					new_target_index = self.dataset.number2idx[new_target]
					social_homophily_sample.extend([source_index, new_target_index])

			negative_social_homophily_samples.append(social_homophily_sample)

		negative_social_homophily_samples = np.array(negative_social_homophily_samples)
		error_message = "wrong negative samples:{}, {}".format(
			negative_social_homophily_samples.shape, (len(positive_samples), 2 * negdup)
		)
		assert negative_social_homophily_samples.shape == (len(positive_samples), 2 * negdup), error_message
		return negative_social_homophily_samples

	def gen_triad_samples(self, positive_samples):
		filtered_positive = [p for p in positive_samples if p[0] + 1 < self.model.timesteps]
		assert len(filtered_positive) > 0, "No possible triangular samples"

		triad_data = []
		while len(triad_data) < len(positive_samples):
			new_samples = []

			for positive_element in filtered_positive:
				new_sample = self.gen_single_triad_sample(positive_element)
				if new_sample != None:
					new_samples.append(new_sample)

			triad_data.extend(new_samples)

		return triad_data[:len(positive_samples)]

	def gen_single_triad_sample(self, positive_element):
		k, source_index, target_index = positive_element
		source = self.dataset.vertices[source_index]
		target = self.dataset.vertices[target_index]
		graph = self.dataset[k]
		next_graph = self.dataset[k + 1]
		trycnt = 0
		ret = None
		filtered_neighbors = []

		if random.randrange(2) == 0:  # target as key point
			neighbors = graph.neighbors(target)

			if len(neighbors) == 0:
				return None

			new_source = random.choice(neighbors)

			while not self.is_triad(source, target, new_source, graph):
				if trycnt >= 5:
					break
				new_source = random.choice(neighbors)
				trycnt += 1

			if trycnt >= 5:
				filtered_neighbors = [n for n in neighbors if self.is_triad(source, target, n, graph)]
				if len(filtered_neighbors) <= 0:
					return None
				new_source = random.choice(filtered_neighbors)

			ret = [
				k,
				target_index,
				source_index,
				self.dataset.number2idx[new_source],
				next_graph.has_edge(source, new_source),
				graph.get_edge_weight(target, source),
				graph.get_edge_weight(target, new_source)
			]
		else:  # src as key point
			neighbors = graph.neighbors(source)

			if len(neighbors) == 0:
				return None

			new_target = random.choice(neighbors)

			while not self.is_triad(source, target, new_target, graph):
				if trycnt >= 5:
					break
				new_target = random.choice(neighbors)
				trycnt += 1

			if trycnt >= 5:
				filtered_neighbors = [n for n in neighbors if self.is_triad(source, target, n, graph)]
				if len(filtered_neighbors) <= 0:
					return None
				new_target = random.choice(filtered_neighbors)

			ret = [
				k,
				source_index,
				target_index,
				self.dataset.number2idx[new_target],
				next_graph.has_edge(target, new_target),
				graph.get_edge_weight(source, target),
				graph.get_edge_weight(source, new_target)
			]

		assert len(set(ret[1:4])) == 3 and ret[5] > 0 and ret[6] > 0, ret
		return ret

	def is_triad(self, i, j, k, graph):
		if i == k:
			return False
		if j == k:
			return False
		if not graph.has_edge(i, k):
			return False
		if not graph.has_edge(j, k):
			return False
		return True

	def calculate_EM_coefficient(self, triad_data):
		embedding = self.model.embedding.detach().to('cpu').numpy()
		theta = self.model.theta.detach().to('cpu').numpy()
		beta = self.model.beta.detach().to('cpu').numpy()

		emcoef_int = []
		emcoef_float = []

		for time_step, k_index, i_index, j_index, has_next_edge, weight_i, weight_j in triad_data:
			C = 1
			partial_embedding = embedding[time_step]
			i, j, k = self.dataset.vertices[i_index], self.dataset.vertices[j_index], self.dataset.vertices[k_index]

			if has_next_edge:
				graph = self.dataset[time_step]

				C0 = self.P(i, i_index, j, j_index, k, k_index, partial_embedding, theta, beta, graph)

				neighbors_i = set(graph.neighbors(i))
				neighbors_j = set(graph.neighbors(j))
				neighbors_common = neighbors_i.intersection(neighbors_j)

				C1 = 1 - np.prod([
					1 - self.P(
						i, i_index,
						j, j_index,
						v, self.dataset.number2idx[v],
						partial_embedding, theta, beta,
						graph
					) for v in neighbors_common
				])
				C = 1 - C0 / (C1 + 1e-6)

			emcoef_int.append([time_step, k_index, i_index, j_index])
			emcoef_float.append([C, weight_i, weight_j])

		return np.array(emcoef_int), np.array(emcoef_float)

	def P(self, a, a_i, b, b_i, c, c_i, partial_embedding, theta, beta, graph):
		w1 = graph.get_edge_weight(a, c)
		w2 = graph.get_edge_weight(b, c)
		c_a = partial_embedding[c_i] - partial_embedding[a_i]
		c_b = partial_embedding[c_i] - partial_embedding[b_i]
		x = c_a * w1 + c_b * w2

		power = -(np.dot(theta, x) + beta)

		if power > 100:
			return 0
		else:
			return 1.0 / (1 + math.exp(power))
