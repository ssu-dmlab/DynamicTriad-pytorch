import torch
import random
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

class Evaluator:
	def __init__(self, mode='link_reconstruction'):
		if mode == 'link_reconstruction':
			self.evaluate = self.evaluate_link_reconstruction
		else:
			raise RuntimeError("Unkonwn task: {}".format(mode))

	def evaluate_link_reconstruction(self, model, dataset):
		samples, labels = self.sample_link_reconstruction(dataset, model.timesteps)

		with torch.no_grad():
			u1 = model.embedding[samples[:, 0], samples[:, 1]]
			u2 = model.embedding[samples[:, 0], samples[:, 2]]
			features = torch.abs(u1 - u2)

		val_scores = []
		cv = StratifiedKFold(n_splits=2, shuffle=True)
		parts = cv.split(features, labels)

		for train, test in parts:
			classifier = LogisticRegression()
			classifier = classifier.fit(features[train], labels[train])
			predict = classifier.predict(features[test])
			val_scores.append(f1_score(labels[test], predict))

		return np.mean(val_scores)

	def sample_link_reconstruction(self, dataset, timesteps, negdup=1):
		positive_samples = []
		negative_samples = []

		for i, graph in enumerate(dataset):
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
