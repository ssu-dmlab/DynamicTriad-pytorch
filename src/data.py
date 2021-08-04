import graph_tools as gt
from loguru import logger

class Dataset():
	def __init__(self, dirname, time):
		self.graphs = []
		self._vertices = set()

		for t in range(time):
			logger.debug("loading {}th time step".format(t))
			filename = '{}/{}'.format(dirname, str(t))
			self.graphs.append(self.load_graph(filename))

		self.vertices = list(self._vertices)
		self.number2idx = {n: i for i, n in enumerate(self.vertices)}

	def __len__(self):
		return self.graphs.__len__()

	def __getitem__(self, idx):
		return self.graphs.__getitem__(idx)

	def load_graph(self, filename):
		graph = gt.Graph(directed=False)
		f = open(filename, 'r')

		lines = f.readlines()
		for line in lines:
			fields = line.split(' ')
			n = int(fields[0])

			if not n in self._vertices:
				self._vertices.add(n)

			if not graph.has_vertex(n):
				graph.add_vertex(n)

			for v, w in zip(fields[1::2], fields[2::2]):
				v = int(v)
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
