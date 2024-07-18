import networkx as nx
from scipy.sparse import csr_matrix
from spektral.layers import GCNConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# class CryptoGCN:
#     def __init__(self, correlation_matrices):
#         self.correlation_matrices = correlation_matrices
#
#     def build_graph(self, correlation_matrix):
#         graph = nx.from_numpy_matrix(correlation_matrix)
#         adjacency_matrix = nx.adjacency_matrix(graph).toarray()
#         return adjacency_matrix
#
#     def graph_convolution(self, sequential_embeddings, adjacency_matrix):
#         X_input = Input(shape=(sequential_embeddings.shape[1],))
#         A_input = Input(shape=(adjacency_matrix.shape[0],))
#
#         gcn_output = GCNConv(16, activation='relu')([X_input, A_input])
#         gcn_output = GCNConv(1, activation='linear')([gcn_output, A_input])
#
#         model = Model(inputs=[X_input, A_input], outputs=gcn_output)
#         model.compile(optimizer='adam', loss='mse')
#         return model
#
#     def apply_gcn(self, sequential_embeddings):
#         gcn_outputs = []
#         for correlation_matrix in self.correlation_matrices:
#             adjacency_matrix = self.build_graph(correlation_matrix)
#             gcn_model = self.graph_convolution(sequential_embeddings, adjacency_matrix)
#             gcn_output = gcn_model.predict([sequential_embeddings, adjacency_matrix])
#             gcn_outputs.append(gcn_output)
#         return gcn_outputs

    #TODO: WORK FROM HERE
