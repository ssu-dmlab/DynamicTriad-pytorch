import graph_tools as gt
from loguru import logger

class Dataset():
	def __init__(self, dirname, time):
		self.graphs = []
		self.vertices = []

		for t in range(time):
			logger.debug("loading {}th time step".format(t))
			filename = '{}/{}'.format(dirname, str(t))
			self.graphs.append(self.load_graph(filename))

		self.number2idx = {n: i for i, n in enumerate(self.vertices)}

	def __len__(self):
		return self.graphs.__len__()

	def __getitem__(self, idx):
		return self.graphs.__getitem__(idx)

	def all_vertices(self):
		return self.vertices

	def number_to_index(self):
		return self.number2idx

	def load_graph(self, filename):
		graph = gt.Graph()
		f = open(filename, 'r')

		lines = f.readlines()
		for line in lines:
			fields = line.split(' ')
			n = int(fields[0])

			if not n in self.vertices:
				self.vertices.append(n)

			if not graph.has_vertex(n):
				graph.add_vertex(n)

			for v, w in zip(fields[1::2], fields[2::2]):
				v = int(v)
				w = float(w)

				if v == n:
					logger.warning("loopback edge ({}, {}) detected".format(v, n))

				if not v in self.vertices:
					self.vertices.append(v)

				if not graph.has_vertex(v):
					graph.add_vertex(v)

				if not graph.has_edge(n, v):
					graph.add_edge(n, v)
					graph.set_edge_weight(n, v, w)

		f.close()
		return graph
