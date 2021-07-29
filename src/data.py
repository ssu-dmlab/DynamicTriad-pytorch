import graph_tools as gt
from loguru import logger

class Dataset():
	def __init__(self, dirname, time):
		self.graphs = []

		for t in range(time):
			filename = '{}/{}'.format(dirname, str(t))
			self.graphs.append(self.load_graph(filename))

	def __len__(self):
		return self.graphs.__len__()

	def __getitem__(self, idx):
		return self.graphs.__getitem__(idx)

	def load_graph(self, filename):
		graph = gt.Graph()
		f = open(filename, 'r')

		lines = f.readlines()
		for line in lines:
			fields = line.split(' ')
			n = int(fields[0])

			if not graph.has_vertex(n):
				graph.add_vertex(n)

			for v, w in zip(fields[1::2], fields[2::2]):
				v = int(v)
				w = float(w)

				if v == n:
					logger.warning("loopback edge ({}, {}) detected".format(v, n))

				if not graph.has_vertex(v):
					graph.add_vertex(v)

				if not graph.has_edge(n, v):
					graph.add_edge(n, v)
					graph.set_edge_weight(n, v, w)

		f.close()
		return graph
