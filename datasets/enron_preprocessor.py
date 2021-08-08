import os
import math
import nltk
import email
import graph_tools
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
import time, dateutil, datetime
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# load data

raw_data = pd.read_csv('emails.csv')
emails = [email.message_from_string(raw_email) for raw_email in tqdm(raw_data['message'], desc='parsing')]

# filter data by time and existence of from, to address

times = [dateutil.parser.parse(e['Date']) for e in tqdm(emails, desc='check time')]
lower_bound = dateutil.parser.parse('1998-11-01 00:00:00 UTC')
upper_bound = dateutil.parser.parse('2002-08-31 23:59:59 UTC')

filtered_times_and_emails = [(t, e) for t, e in zip(times, emails) if lower_bound < t and t < upper_bound]
filtered_times_and_emails = [(t, e) for t, e in filtered_times_and_emails if e['From'] is not None and e['To'] is not None]
filtered_times = [datetime.date(t.year, t.month, 1) for t, _ in filtered_times_and_emails]
filtered_emails = [e for _, e in filtered_times_and_emails]

# get document embedding using nlp

embedding_dimension = 128

nltk.download('punkt')

corpus = [e.get_payload() for e in tqdm(filtered_emails, desc='loading payload')]
corpus = [nltk.word_tokenize(document.lower()) for document in tqdm(corpus, desc='tokenize')]
corpus = [TaggedDocument(words, [i]) for i, words in tqdm(enumerate(corpus), total=len(corpus), desc='tagging')]

model = Doc2Vec(vector_size=embedding_dimension, window=5)
model.build_vocab(corpus)
for _ in tqdm(range(10), desc='epoch', position=0):
	model.train(tqdm(corpus, desc='train', position=1), epochs=1, total_examples=model.corpus_count)

vectors = [model.infer_vector(document.words) for document in tqdm(corpus, desc='infer vector')]

# generate edges for dynamic graph

spilt_data = {}

min_time = min(filtered_times)
max_time = max(filtered_times)

offset = min_time.year * 12 + min_time.month
end = max_time.year * 12 + max_time.month

edges_by_time = [[] for _ in range(end - offset + 1)]

for t, e, v in zip(filtered_times, filtered_emails, vectors):
	i = t.year * 12 + t.month - offset

	froms = e['From']
	froms = froms.replace(',', ' ')
	froms = froms.replace('\t', ' ')
	froms = froms.replace('\n', ' ')
	while froms.find('  ') != -1:
		froms = froms.replace('  ', ' ')
	froms = froms.split(' ')

	tos = e['To']
	tos = tos.replace(',', ' ')
	tos = tos.replace('\t', ' ')
	tos = tos.replace('\n', ' ')
	while tos.find('  ') != -1:
		tos = tos.replace('  ', ' ')
	tos = tos.split(' ')

	for f in froms:
		for t in tos:
			edges_by_time[i].append({'From':f, 'To': t, 'Feature':v})

# generate directed dynamic graph

vertices = set()
graphs = [graph_tools.Graph() for _ in edges_by_time]

for i, edges in enumerate(edges_by_time):
	graph = graphs[i]

	for edge in edges:
		f, t, v = edge['From'], edge['To'], edge['Feature']

		if f is None or t is None:
			logger.warning('vertex is none')

		if not f in vertices:
			vertices.add(f)

		if not graph.has_vertex(f):
			graph.add_vertex(f)

		if not t in vertices:
			vertices.add(t)

		if not graph.has_vertex(t):
			graph.add_vertex(t)

		if graph.has_edge(f, t):
			graph.set_edge_weight(f, t, graph.get_edge_weight(f, t) + 1.0)
			prev_features = graph.get_edge_attribute_by_id(f, t, 0, 'features')
			graph.set_edge_attribute_by_id(f, t, 0, 'features', prev_features + [v])
		else:
			graph.add_edge(f, t)
			graph.set_edge_weight(f, t, 1.0)
			graph.set_edge_attribute_by_id(f, t, 0, 'features', [v])

# add edge embedding

vertices = list(vertices)
vertex2index = {n: i for i, n in enumerate(vertices)}

embedding = []

for graph in graphs:
	embedding_on_time = []

	for vertex in vertices:
		if graph.has_vertex(vertex):
			edges = graph.edges_from(vertex)
			if edges != []:
				features = []
				for f, t in edges:
					features.extend(graph.get_edge_attribute_by_id(f, t, 0, 'features'))
				features = np.array(features)
				embedding_on_time.append(np.mean(features, axis=0))
			else:
				embedding_on_time.append(None)
		else:
			embedding_on_time.append(None)

	embedding.append(embedding_on_time)

# set default embedding to mean of all embedding
# WARNING: this is a naive solution!

not_nones = []

for embedding_on_time in embedding:
	for v in embedding_on_time:
		if v is not None:
			not_nones.append(v)

mean_vector = np.mean(np.array(not_nones), axis=0)

for t, embedding_on_time in enumerate(embedding):
	for i, v in enumerate(embedding_on_time):
		if v is None:
			embedding[t][i] = mean_vector

# generate undirected graph

undirected_graphs = []

for t, graph in enumerate(graphs):
	undirected_graph = graph_tools.Graph(directed=False)

	for vertex in graph.vertices():
		undirected_graph.add_vertex(vertex)
		vertex_index = vertex2index[vertex]
		undirected_graph.set_vertex_attribute(vertex, 'feature', embedding[t][vertex_index])

	for u, v in graph.edges():
		if undirected_graph.has_edge(u, v):
			prev_weight = undirected_graph.get_edge_weight(u, v)
			new_weight = prev_weight + graph.get_edge_weight(u, v)
			undirected_graph.set_edge_weight(u, v, prev_weight)
		else:
			undirected_graph.add_edge(u, v)
			weight = graph.get_edge_weight(u, v)
			undirected_graph.set_edge_weight(u, v, weight)

	undirected_graphs.append(undirected_graph)

# save graphs into files

os.makedirs('enron', exist_ok=True)

for t, undirected_graph in tqdm(enumerate(undirected_graphs), total=len(undirected_graphs), desc='save graphs'):
	with open('enron/' + str(t), 'w') as file:
		for vertex in undirected_graph.vertices():
			file.write(vertex)
			for neighbor in undirected_graph.neighbors(vertex):
				if vertex < neighbor:
					weight = undirected_graph.get_edge_weight(vertex, neighbor)
					file.write(' ' + neighbor + ' ' + str(weight))
			file.write('\n')

	with open('enron/' + str(t) + '.feature', 'w') as file:
		file.write(str(embedding_dimension) + '\n')
		for vertex in vertices:
			file.write(vertex)
			vector = embedding[t][vertex2index[vertex]]
			for feature in vector:
				file.write(' ' + str(feature))
			file.write('\n')
