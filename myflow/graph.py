class Graph(object):
    def __init__(self):
        self.TRAIN_VARS_COLLECTIONS = []
        self.CONSTANTS_COLLECTIONS = []
        self.PLACEHOLDER_COLLECTIONS = []
        self.OPERATION_TENSORS_COLLECTIONS = []

    def as_default(self, graph):
        Graph.default_graph = graph

    @classmethod
    def get_default_graph(cls):
        return cls.default_graph

    def __enter__(self):
        Graph.default_graph = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Graph.default_graph = default_graph


default_graph = Graph()
Graph.default_graph = default_graph