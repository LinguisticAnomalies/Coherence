import networkx as nx
import pandas as pd
import numpy as np
from coherencecalculator.tools.vecloader import VecLoader



class SpeechGraph(object):
    def __init__(self, vecLoader:VecLoader):
        self.nlp = vecLoader.nlp

    
    def create_graph(self, text:str, window_size=2) -> nx.multidigraph:
        # Process the text
        doc = self.nlp(text)
        
        #print(f"--- sliding window: {window_size} ---")
        G = nx.MultiDiGraph()
        valid_tokens = [token for token in doc if not token.is_punct and not token.is_space]
        #print(f"unique word count: {len(set([token.text for token in valid_tokens]))}")

        for i in range(len(valid_tokens) - window_size + 1):
            window = [token.text for token in valid_tokens[i:i+window_size]]
            if len(window) > 1:
                edges = list(zip(window[:-1], window[1:]))
                G.add_edges_from(edges)

        return G
    def get_number_of_nodes(self, graph:nx.multidigraph) -> int:
        return graph.number_of_nodes()
    
    def get_number_of_edges(self, graph:nx.multidigraph) -> int:
        return graph.number_of_edges()
    
    def get_paralle_edges(self, graph:nx.multidigraph) -> int:
        # Custom edge counting
        edge_counts = {}
        for u, v in graph.edges():
            edge_key = (u, v)
            edge_counts[edge_key] = len(graph.get_edge_data(u,v))
            
        return sum(1 for count in edge_counts.values() if count > 1)
    
    def get_number_of_scc(self, graph:nx.multidigraph) -> int:
        return nx.number_strongly_connected_components(graph)
    
    def get_number_of_nodes_lsc(self, graph:nx.multidigraph) -> int:
        return len(max(nx.strongly_connected_components(graph), key=len))
    
    def get_density(self, graph:nx.multidigraph) -> float:
        return nx.density(graph)
    
    def get_avg_degree(self, graph:nx.multidigraph) -> float:
        degrees = list(dict(graph.degree()).values())
        return np.mean(degrees), np.std(degrees)
    
    def get_self_loop(self, graph:nx.multidigraph) -> int:
        adj_matrix = nx.linalg.adjacency_matrix(graph).toarray()
        # Calculate number of self-loops
        return int(np.trace(adj_matrix))
    
    def add_speechgraph_features(self, data:pd.DataFrame, window_size=2) -> pd.DataFrame:
        result = data.copy()
        result['number_of_nodes'] = result['number_of_edges'] = result['PE'] = result['number_scc'] = None
        result['LSC'] = result['density'] = result['degree_average'] = result['degree_std'] = result['L1'] = None
        for i, row in result.iterrows():
            text = row['text']
            if type(text) == list:
                text = ' '.join(text)
            graph = self.create_graph(text, window_size)
            n_nodes = self.get_number_of_nodes(graph)
            if n_nodes == 0:#Protection against empty graph exceptions
                result.at[i, 'number_of_nodes'] = [0]
                result.at[i, 'number_of_edges'] = [0]
                result.at[i, 'PE'] = [0]
                result.at[i, 'number_scc'] = [0]                
                result.at[i, 'LSC'] = [0]                
                result.at[i, 'density'] = [0]
                result.at[i, 'degree_average'] = [0]
                result.at[i, 'degree_std'] = [0]
                result.at[i, 'L1'] = [0]
            else:
                result.at[i, 'number_of_nodes'] = [n_nodes]
                result.at[i, 'number_of_edges'] = [self.get_number_of_edges(graph)]
                result.at[i, 'PE'] = [self.get_paralle_edges(graph)]
                
                
                # Calculate number of nodes in the maximum connected component
                result.at[i, 'number_scc'] = [self.get_number_of_scc(graph)]
                
                # Calcualte number of nodes in the maximum strongly connected componet
                result.at[i, 'LSC'] = [self.get_number_of_nodes_lsc(graph)]
                
                result.at[i, 'density'] = [self.get_density(graph)]
                
                degree_avg, degree_std = self.get_avg_degree(graph)
                result.at[i, 'degree_average'] = [degree_avg]
                result.at[i, 'degree_std'] = [degree_std]
            
                # Calculate number of self-loops
                result.at[i, 'L1'] = [self.get_self_loop(graph)]
        
        return result
    

        
    # def analyze_dependency_graph(self, text, window_size=2):
        
    #     # Process the text
    #     doc = self.nlp(text)
        
    #     #print(f"--- sliding window: {window_size} ---")
    #     G = nx.MultiDiGraph()
    #     valid_tokens = [token for token in doc if not token.is_punct and not token.is_space]
    #     #print(f"unique word count: {len(set([token.text for token in valid_tokens]))}")

    #     for i in range(len(valid_tokens) - window_size + 1):
    #         window = [token.text for token in valid_tokens[i:i+window_size]]
    #         if len(window) > 1:
    #             edges = list(zip(window[:-1], window[1:]))
    #             G.add_edges_from(edges)

    #     stats = calculate_statistics(G)
    #     for k, v in stats.items():
    #         print(f"{k}\t{round(v, 3)}")

    #     # plt.figure(figsize=(12, 10))
    #     # nx.draw(G, with_labels=True, font_size=6, font_weight='light')
    #     # plt.savefig("../data/sample_graph.png", dpi=300, bbox_inches='tight')
    #     return G
        

    # def calculate_statistics(graph):
    #     res = {}
    #     res['number_of_nodes'] = graph.number_of_nodes()
    #     res['number_of_edges'] = graph.number_of_edges()
        
    #     # Custom edge counting
    #     edge_counts = {}
    #     for u, v in graph.edges():
    #         edge_key = (u, v)
    #         edge_counts[edge_key] = len(graph.get_edge_data(u,v))
        
    #     # Calculate parallel edges
    #     res['PE'] = sum(1 for count in edge_counts.values() if count > 1)
    #     # Calculate number of nodes in the maximum connected component
    #     res['number_scc'] = len(list(nx.strongly_connected_components(graph)))
    #     # Calcualte number of nodes in the maximum strongly connected componet
    #     res['LSC'] = len(max(nx.strongly_connected_components(graph), key=len))
    #     res['number_of_cycles'] = len(list(nx.simple_cycles(graph, length_bound=2)))
    #     print(list(nx.simple_cycles(graph, length_bound=2)))
    #     # scc = list(nx.strongly_connected_components(graph))
    #     # for item in scc:
    #     #     print(item)
        
    #     degrees = list(dict(graph.degree()).values())
    #     res['density'] = nx.density(graph)
    #     res['degree_average'] = np.mean(degrees)
    #     res['degree_std'] = np.std(degrees)
        
    #     adj_matrix = nx.linalg.adjacency_matrix(graph).toarray()
    #     # Calculate number of self-loops
    #     res['L1'] = int(np.trace(adj_matrix))

    #     return res