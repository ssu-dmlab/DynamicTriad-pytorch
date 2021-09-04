import torch
import random
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# this code refers https://github.com/luckiezhou/DynamicTriad/blob/master/scripts/stdtests.py
# this code refers https://github.com/luckiezhou/DynamicTriad/blob/master/core/algorithm/embutils.py
# this code refers https://github.com/luckiezhou/DynamicTriad/blob/master/core/dataset/dataset_utils.py

class Evaluator:
	def __init__(self, mode='link_reconstruction'):
		if mode == 'link_reconstruction':
			self.evaluate = self.evaluate_link_reconstruction
		elif mode == 'link_prediction':
			self.evaluate = self.evaluate_link_prediction
		elif mode == 'change_link_reconstruction':
			self.evaluate = self.evaluate_change_link_reconstruction
		elif mode == 'change_link_prediction':
			self.evaluate = self.evaluate_change_link_prediction
		else:
			raise RuntimeError("Unkonwn task: {}".format(mode))

	def evaluate_link_reconstruction(self, model, dataset, interval=0):
		samples, labels = self.sample_link_reconstruction(dataset, model.timesteps, interval=interval)

		with torch.no_grad():
			u1 = model.embedding[samples[:, 0], samples[:, 1]]
			u2 = model.embedding[samples[:, 0], samples[:, 2]]
			features = torch.abs(u1 - u2).cpu()

		val_scores = []
		cv = StratifiedKFold(n_splits=5, shuffle=True)
		parts = cv.split(features, labels)

		for train, test in parts:
			classifier = LogisticRegression()
			classifier = classifier.fit(features[train], labels[train])
			predict = classifier.predict(features[test])
			val_scores.append(f1_score(labels[test], predict))

		return np.mean(val_scores)

	def sample_link_reconstruction(self, dataset, timesteps, interval=0, negdup=1):
		positive_samples = []
		negative_samples = []

		for i, graph in enumerate(dataset[interval:]):
			if i >= timesteps:
				break

			for e in graph.edges():
				e0_index = dataset.vertex2index[e[0]]
				e1_index = dataset.vertex2index[e[1]]
				positive_samples.append([i, e0_index, e1_index])

		for _ in range(negdup):
			for i, source_index, target_index in positive_samples:
				graph = dataset[i]
				source = dataset.vertices[source_index]
				target = dataset.vertices[target_index]

				while True:
					if random.randrange(2) == 0: # replace source
						new_source_index = random.randrange(len(dataset.vertices))
						new_source = dataset.vertices[new_source_index]
						if not graph.has_edge(new_source, target):
							negative_samples.append([i, new_source_index, target_index])
							break
					else: # replace target
						new_target_index = random.randrange(len(dataset.vertices))
						new_target = dataset.vertices[new_target_index]
						if not graph.has_edge(new_target, source):
							negative_samples.append([i, source_index, new_target_index])
							break

		positive_samples = np.array(positive_samples)
		negative_samples = np.array(negative_samples)
		samples = np.concatenate((positive_samples, negative_samples))

		positive_labels = np.ones(positive_samples.shape[0])
		negative_labels = np.zeros(negative_samples.shape[0])
		labels = np.concatenate((positive_labels, negative_labels))

		return samples, labels

	def evaluate_link_prediction(self, model, dataset):
		return self.evaluate_link_reconstruction(model, dataset, interval=1)

	def evaluate_change_link_reconstruction(self, model, dataset, interval=0):
		samples, labels = self.sample_change_link_reconstruction(dataset, model.timesteps, interval=interval)

		with torch.no_grad():
			u1 = model.embedding[samples[:, 0] - interval, samples[:, 1]]
			u2 = model.embedding[samples[:, 0] - interval, samples[:, 2]]
			features = torch.abs(u1 - u2).cpu()

		val_scores = []
		cv = StratifiedKFold(n_splits=5, shuffle=True)
		parts = cv.split(features, labels)

		for train, test in parts:
			classifier = LogisticRegression()
			classifier = classifier.fit(features[train], labels[train])
			predict = classifier.predict(features[test])
			val_scores.append(f1_score(labels[test], predict))

		return np.mean(val_scores)

	def sample_change_link_reconstruction(self, dataset, timesteps, interval=0):
		prev_graphs = dataset[interval:len(dataset)-1]
		current_graphs = dataset[interval+1:]

		samples, labels = [], []

		for i, (prev_graph, current_graph) in enumerate(zip(prev_graphs, current_graphs)):
			if i >= timesteps-1:
				break

			prev_edges = set()
			for e0, e1 in prev_graph.edges():
				if not (e0, e1) in prev_edges:
					prev_edges.add((e0, e1))

			current_edges = set()
			for e0, e1 in current_graph.edges():
				if not (e0, e1) in current_edges:
					current_edges.add((e0, e1))

			for e0, e1 in current_edges - prev_edges:
				e0 = dataset.vertex2index[e0]
				e1 = dataset.vertex2index[e1]
				samples.append([i+1, e0, e1])
				labels.append(1)

			for e0, e1 in prev_edges - current_edges:
				e0 = dataset.vertex2index[e0]
				e1 = dataset.vertex2index[e1]
				samples.append([i+1, e0, e1])
				labels.append(0)

		return np.array(samples), np.array(labels)

	def evaluate_change_link_prediction(self, model, dataset):
		return self.evaluate_change_link_reconstruction(model, dataset, interval=1)
