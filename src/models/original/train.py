import math
import torch
import random
import numpy as np
from tqdm import tqdm
from models.original.model import Model

class Trainer:
	def __init__(self, model, dataset):
		self.model = model
		self.dataset = dataset

	def train(self, lr=0.1, epochs=10, batchsize=1000, negdup=1):
		optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		positive_samples, weight = self.gen_positive_samples()

		pbar = tqdm(range(epochs), position=0, leave=False, desc='epoch')
		for epoch in pbar:
			negative_social_homophily_samples = self.gen_social_homophily_samples(positive_samples)
			triad_data = self.gen_triad_samples(positive_samples)
			emcoef_int, emcoef_float = self.calculate_EM_coefficient(triad_data)

			data = []
			for i in range(len(positive_samples)):
				for j in range(0, len(negative_social_homophily_samples[i]), 2):
					data.append(list(pos[i]) + [negative_social_homophily_samples[i][j], negative_social_homophily_samples[i][j+1]])

			ave_loss = 0
			total_batches = math.ceil(len(positive_samples/batchsize))

			pbar2 = tqdm(
				self.gen_batches(positive_samples, weight, emcoef_int, emcoef_float, bacthsize),
				position=1, leave=False, desc='batch'
			)
			for batch in pbar2:
				positive_sample_batch, weight_batch, emcoef_int_batch, emcoef_float_batch = batch

				optimizer.zero_grad()
				loss = self.model(
					positive_sample_batch,
					weight_batch,
					emcoef_int_batch,
					emcoef_float_batch
				)
				loss.backward()
				optimizer.step()

				ave_loss += loss / total_batches

			pbar.write('Epoch {:02}: {:.4} training loss'.format(epoch, loss.item()))
			pbar.update()

		pbar.close()

		return self.model

	def gen_batches(self, positive_samples, weight, emcoef_int, emcoef_float, batchsize):
		curstart = 0
		while True:
			if curstart >= len(positive_samples):
				break

			positive_sample_batch = positive_samples[curstart:min(len(positive_samples), curstart+batchsize)]
			weight_batch = weight[curstart:min(len(weight), curstart+batchsize)]
			emcoef_int_batch = emcoef_int[curstart:min(len(emcoef_int), curstart+batchsize)]
			emcoef_float_batch = emcoef_float[curstart:min(len(emcoef_float), curstart+batchsize)]

			yield positive_sample_batch, weight_batch, emcoef_int_batch, emcoef_float_batch
			curstart += batch_size

	def gen_positive_samples(self):
		positive_samples, weight = [], []

		for i, graph in enumerate(self.dataset):
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
		return data, weight

	def gen_social_homophily_samples(self, positive_samples, negdup=1):
		negative_social_homophily_samples = []

		for k, source_index, target_index in positive_samples:
			source = self.dataset.vertices[source_index]
			target = self.dataset.vertices[target_index]
			graph = self.dataset[k]
			social_homophily_sample = []

			for _ in range(negdup):
				if random.randrange(2) == 0:  # replace source
					new_source = random.randrange(len(graph.V))
					while graph.has_edge(new_source, target):
						new_source = random.randrange(len(graph.V))
					new_source_index = self.dataset.number2idx[new_source]
					social_homophily_sample.extend([new_source_index, target_index])

				else:  # replace target
					new_target = random.randrange(len(graph.V))
					while graph.has_edge(source, new_target):
						new_target = random.randrange(len(graph.V))
					new_target_index = self.dataset.number2idx[new_targe]
					social_homophily_sample.extend([source_index, new_target_index])

			negative_social_homophily_samples.append(social_homophily_sample)

		negative_social_homophily_samples = np.array(negative_social_homophily_samples)
		error_message = "wrong negative samples:{}, {}".format(
			negative_social_homophily_samples.shape, (len(positive_samples), 2 * negdup)
		)
		assert negative_social_homophily_samples.shape == (len(posdata), 2 * negdup), error_message
		return negative_social_homophily_samples

	def gen_triad_samples(self, positive_samples):
		filtered_positive = [p for p in positive_samples if p[0] + 1 < len(self.dataset)]
		assert len(filtered_positive) > 0, "No possible triangular samples"

		triad_data = []
		while len(triad_data) < len(positive_samples):
			left_count = len(positive_samples) - len(triad_data)

			left_count *= 1.2 # increase the probability of finish sampling in a single round

			lower_bound = max(0, len(filtered_positive) - left_count)
			upper_bound = min(lower_bound + left_count, len(filtered_positive))

			new_samples = []
			for positive_element in filtered_positive[lower_bound:upper_bound]:
				new_sample = self.gen_single_triad_sample(positive_element)
				new_samples.append(new_sample)

			triad_data.extend(newsamples)

		return triad_data[:len(positive_samples)]

	def gen_single_triad_sample(self, positive_element):
		k, source, target = positive_element
		graph = self.dataset[k]
		next_graph = self.dataset[k + 1]
		trycnt = 0
		ret = None

		if utils.crandint(2) == 0:  # target as key point
			neighbours = myg.out_neighbours(target)
			new_source = neighbours[utils.crandint(len(neighbours))]

			while new_source == target or new_source == source or not graph.has_edge(target, new_source) or not graph.has_edge(source, new_source):
				if trycnt >= 5:
					break
				new_source = neighbours[utils.crandint(len(neighbours))]
				trycnt += 1

			if trycnt >= 5:
				filtered_neighbours = [n for n in filtered_neighbours if n != source and n != target and not graph.has_edge(n, source)]
				if len(filtered_neighbours) <= 0:
					return None
				new_source = filtered_neighbours[utils.crandint(len(filtered_neighbours))]

			ret = [
				k,
				self.dataset.number2idx[target],
				self.dataset.number2idx[source],
				self.dataset.number2idx[new_source],
				next_graph.has_edge(source, new_source),
				graph.get_edge_weight(target, source),
				graph.get_edge_weight(target, new_source)
			]
		else:  # src as key point
			neighbours = myg.out_neighbours(source)
			new_target = neighbours[utils.crandint(len(neighbours))]

			while new_target == source or new_target == target or not graph.has_edge(source, new_target) or not graph.has_edge(target, new_target):
				if trycnt >= 5:
					break
				new_target = neighbours[utils.crandint(len(neighbours))]
				trycnt += 1

			if trycnt >= 5:
				filtered_neighbours = [n for n in filtered_neighbours if n != target and n != source and not graph.has_edge(n, target)]
				if len(filtered_neighbours) <= 0:
					return None
				new_target = filtered_neighbours[utils.crandint(len(filtered_neighbours))]

			ret = [
				k,
				self.dataset.number2idx[source],
				self.dataset.number2idx[target],
				self.dataset.number2idx[new_target],
				next_graph.has_edge(target, new_target),
				graph.get_edge_weight(source, target),
				graph.get_edge_weight(source, new_target)
			]

		assert len(set(ret[1:4])) == 3 and ret[5] > 0 and ret[6] > 0, ret
		return ret

	def calculate_EM_coefficient(self, triad_data):
		embedding = self.model.embedding.detach().to('cpu').numpy()
		theta = self.model.theta.detach().to('cpu').numpy()
		beta = self.model.beta.detach().to('cpu').numpy()

		emcoef_int = []
		emcoef_float = []

		for time_step, k_index, i_index, j_index, has_next_edge, weight_i, weight_j in triad_data:
			C = 1
			partial_embedding = embedding[timestep]
			i, j, k = self.dataset.vertices[i_index], self.dataset.vertices[j_index], self.dataset.vertices[k_index]

			if has_next_edge:
				graph = selt.dataset[time_step]

				C0 = self.P(i, j, k, partial_embedding, theta, beta, graph)

				neighbours_i = set(graph.out_neighbours(i))
				neighbours_j = set(graph.out_neighbours(j))
				neighbours_common = neighbours_i.intersection(neighbours_j)

				C1 = 1 - np.prod([1 - self.P(i, j, v, partial_embedding, theta, beta, graph) for v in cmnbr])
				C = 1 - C0 / (C1 + 1e-6)

			emcoef_int.append([time_step, k_index, i_index, j_index])
			emcoef_float.append([C, weight_i, weight_j])

		return emcoef_int, emcoef_float

	def P(self, a, b, c, partial_embedding, theta, beta, graph):
		w1 = graph.get_edge_weight(a, c)
		w2 = graph.get_edge_weight(b, c)
		x = (partial_embedding[c] - partial_embedding[a]) * w1 + (partial_embedding[c] - partial_embedding[b]) * w2

		power = -np.dot(theta, x) + beta

		if power > 100:
			return 0
		else:
			return 1.0 / (1 + math.exp(power))
