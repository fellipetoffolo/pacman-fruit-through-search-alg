import re

class Graph:
    def __init__(self):
        self.edges = {}

    def add_edge(self, src, dest, weight):
        if src in self.edges:
            self.edges[src].append((dest, weight))
        else:
            self.edges[src] = [(dest, weight)]

    def add_extremes(self, first, last):

        self.edges['estado inicial'] = first
        self.edges['estado final'] = last

    def get_neighbors(self, node):
        return self.edges.get(node, []) # Método get retorna corretamente para o caso de lista vazia

def readGraphFromFile(filename, graph):
    
    try:
        with open(filename, 'r') as file:  # Abre o arquivo em modo de leitura
            content = file.read()  # Lê todo o conteúdo do arquivo
            
         # Regex para extrair pontos inicial e final
        ponto_inicial = re.search(r"ponto_inicial\((\w+)\)\.", content)
        ponto_final = re.search(r"ponto_final\((\w+)\)\.", content)

        # Regex para extrair transições e custos
        transitions = re.findall(r"pode_ir\((\w+),(\w+),(\d+)\)\.", content)

        # Processa os resultados
        start = ponto_inicial.group(1) if ponto_inicial else None
        end = ponto_final.group(1) if ponto_final else None
        edges = [(from_node, to_node, int(cost)) for from_node, to_node, cost in transitions]

        graph.add_extremes(start, end)
        for src, dest, weight in edges:
            graph.add_edge(src, dest, weight)
    
    except FileNotFoundError:
        print("O arquivo especificado não foi encontrado.")
        return None, None
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None, None
    
graph = Graph()

readGraphFromFile('sample.txt', graph)

print(graph.edges)
