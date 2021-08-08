import graph_tools as gt
from loguru import logger
import numpy as np

class Dataset():
	def __init__(self, dirname, time, load_feature=False):
		self.graphs = []
		self._vertices = set()

		for t in range(time):
			filename = '{}/{}'.format(dirname, str(t))
			self.graphs.append(self.load_graph(filename))

		self.vertices = list(self._vertices)
		self.vertex2index = {n: i for i, n in enumerate(self.vertices)}

		self.feature_dimension = None

		if load_feature:
			for t in range(time):
				graph = self.graphs[t]
				filename = '{}/{}.feature'.format(dirname, t)
				features, dimension = self.load_feature(filename)

				if self.feature_dimension is None:
					self.feature_dimension = dimension
				else:
					assert self.feature_dimension == dimension

				for vertex, feature in features.items():
					if not graph.has_vertex(vertex):
						graph.add_vertex(vertex)
					graph.set_vertex_attribute(vertex, 'feature', feature)

	def __len__(self):
		return self.graphs.__len__()

	def __getitem__(self, idx):
		return self.graphs.__getitem__(idx)

	def load_graph(self, filename):
		graph = gt.Graph(directed=False)
		f = open(filename, 'r')

		for line in f.readlines():
			fields = line.split(' ')
			n = fields[0]

			if not n in self._vertices:
				self._vertices.add(n)

			if not graph.has_vertex(n):
				graph.add_vertex(n)

			for v, w in zip(fields[1::2], fields[2::2]):
				w = float(w)

				if v == n:
					logger.warning("loopback edge ({}, {}) detected".format(v, n))

				if not v in self._vertices:
					self._vertices.add(v)

				if not graph.has_vertex(v):
					graph.add_vertex(v)

				if not graph.has_edge(n, v):
					graph.add_edge(n, v)
					graph.set_edge_weight(n, v, w)

		f.close()
		return graph

	def load_feature(self, filename):
		file = open(filename, 'r')
		dimension = int(file.readline())
		lines = file.readlines()
		features = {}

		for line in file.readlines():
			fields = line.split(' ')
			vertex = fields[0]
			feature = [float(f) for f in fields[1:]]
			assert len(feature) == dimension
			feature = np.array(feature)
			features[vertex] = feature

		file.close()
		return features, dimension
