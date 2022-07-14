# %%
from IPython.display import SVG
import numpy as np
from sknetwork.data import karate_club, painters, movie_actor
from sknetwork.classification import KNN
from sknetwork.embedding import GSVD
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph
# %%
graph = karate_club(metadata=True)
adjacency = graph.adjacency
position = graph.position
labels_true = graph.labels
# %%
seeds = {i: labels_true[i] for i in [0, 33]}
print(seeds)
# %%
knn = KNN(GSVD(3), n_neighbors=1)
labels_pred = knn.fit_transform(adjacency, seeds)
print(labels_pred)
# %%
precision = np.round(np.mean(labels_pred == labels_true), 2)
precision
# %%
image = svg_graph(adjacency, position, labels=labels_pred, seeds=seeds)
SVG(image)
# %%
knn = KNN(GSVD(3), n_neighbors=2)
knn.fit(adjacency, seeds)
membership = knn.membership_
print(membership)
# %%
scores = membership[:,1].toarray().ravel()
print(scores)
# %%
image = svg_graph(adjacency, position, scores=scores, seeds=seeds)
SVG(image)
# %%
graph = painters(metadata=True)
adjacency = graph.adjacency
position = graph.position
names = graph.names
# %%
rembrandt = 5
klimt = 6
cezanne = 11
seeds = {cezanne: 0, rembrandt: 1, klimt: 2}
print(seeds)
# %%
knn = KNN(GSVD(3), n_neighbors=2)
labels = knn.fit_transform(adjacency, seeds)
print(labels)
# %%
