import re
from collections import deque #bfs e bidirecional
import heapq #ucs


class Graph:
    def __init__(self):
        self.edges = {}
        self.heuristic = {}

    def add_edge(self, src, dest, weight):
        if src in self.edges:
            self.edges[src].append((dest, weight))
        else:
            self.edges[src] = [(dest, weight)]

    def add_extremes(self, first, last):

        self.edges['estado inicial'] = first
        self.edges['estado final'] = last

    def add_heuristic(self, state, value):

        self.heuristic[state] = value

    def get_neighbors(self, node):
        return self.edges.get(node, [])

def format_graph(graph):
    formated_graph = "grafo formatado: \n"
    for src, dest in graph.edges.items():
        if src != "estado inicial" and src != "estado final":
            for vertex in dest:
                formated_graph = formated_graph + src + "->" + vertex[0] + "\n"

    formated_graph = formated_graph + "Valores da heurística: \n"

    for state, value in graph.heuristic.items():
        formated_graph = formated_graph + state + ", " + str(value) + "\n"
    return formated_graph

def read_graph_from_file(filename, graph):

    try:
        with open(filename, 'r') as file:  # Abre o arquivo em modo de leitura
            content = file.read()  # Lê todo o conteúdo do arquivo

        # Regex para extrair pontos inicial e final
        ponto_inicial = re.search(r"ponto_inicial\s*\(\s*(\w+)\s*\)\s*\.?", content)
        ponto_final = re.search(r"ponto_final\s*\(\s*(\w+)\s*\)\s*\.?", content)

        # Regex para extrair transições e custos
        transitions = re.findall(r"pode_ir\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*\)\s*\.?", content)

        # Regex para extrair a heurística
        heuristics = re.findall(r"h\s*\(\s*(\w+)\s*,\s*\w+\s*,\s*(\d+)\s*\)\s*\.?", content)

        # Processa os resultados
        start = ponto_inicial.group(1) if ponto_inicial else None
        end = ponto_final.group(1) if ponto_final else None
        edges = [(from_node, to_node, int(cost)) for from_node, to_node, cost in transitions]

        graph.add_extremes(start, end)
        for src, dest, weight in edges:
            graph.add_edge(src, dest, weight)

        heuristics_dict = {state : int(value) for state, value in heuristics}

        for state, value in heuristics_dict.items():
            graph.add_heuristic(state, value)

    except FileNotFoundError:
        print("O arquivo especificado não foi encontrado.")
        return None, None
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None, None

########################################################################
# Funções de medida de desemepenho
########################################################################

def desempenhoNos(visited_nodes_count):
    """
    Monitora o número de nós visitados.
    
    :param visited_nodes_count: Contador de nós visitados.
    :return: O contador de nós visitados atualizado.
    """
    visited_nodes_count += 1
    print(f"Nós visitados: {visited_nodes_count}")
    return visited_nodes_count

def desempenhoFronteira(frontier, max_frontier_size):
    """
    Monitora o tamanho atual e máximo da fronteira.
    
    :param frontier: A estrutura que contém os nós da fronteira.
    :param max_frontier_size: O tamanho máximo da fronteira observado até agora.
    :return: O tamanho máximo atualizado da fronteira.
    """
    current_frontier_size = len(frontier)
    if current_frontier_size > max_frontier_size:
        max_frontier_size = current_frontier_size
    print(f"Tamanho atual da fronteira: {current_frontier_size}")
    print(f"Tamanho máximo da fronteira observado: {max_frontier_size}")
    
    return max_frontier_size

########################################################################
# Funções dos algoritmos
########################################################################

def greedy_search(graph, heuristic):
    """
    Implementação da Busca Gulosa (Greedy Search).
    
    :param graph: Instância da classe Graph.
    :param start: O nó de partida para a busca.
    :param goal: O nó objetivo que queremos alcançar.
    :param heuristic: Um dicionário que contém a estimativa (heurística) do custo para alcançar o objetivo de cada nó.
    :return: Uma lista representando o caminho do nó inicial ao nó objetivo, se encontrado.
    """

    start = graph.edges['estado inicial']
    goal = graph.edges['estado final']
    # Inicializa a fronteira (open list) com o nó de início
    frontier = [(start, 0)]  # A fronteira é uma lista de tuplas (nó, custo)
    
    # Dicionário para rastrear o caminho
    came_from = {}
    came_from[start] = None

    visited_nodes_count = 0
    max_frontier_size = 0
    iteration = 1
    # Enquanto houver nós na fronteira para explorar
    while frontier:
        print(f"\nIteração {iteration}")
        iteration += 1
        # Atualiza o desempenho da fronteira
        max_frontier_size = desempenhoFronteira(frontier, max_frontier_size)

        # Ordena a fronteira pela heurística (a fronteira é uma lista de tuplas)
        frontier.sort(key=lambda x: heuristic[x[0]])
        
        # Escolhe o nó com a menor heurística
        current_node, current_cost = frontier.pop(0)

        # Atualiza o desempenho dos nós visitados
        visited_nodes_count = desempenhoNos(visited_nodes_count)
        
        # Se o nó atual é o objetivo, reconstruir o caminho e retornar
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]

            #Reporta os valores de desempenho finais
            print(f"\n\nTotal de nós visitados: {visited_nodes_count}")
            print(f"Tamanho máximo da fronteira: {max_frontier_size}")
            return path[::-1]  # Retorna o caminho do início ao objetivo
        
        # Explora os vizinhos do nó atual
        for neighbor, cost in graph.get_neighbors(current_node):
            if neighbor not in came_from:  # Se o vizinho ainda não foi explorado
                frontier.append((neighbor, cost))  # Adiciona o vizinho à fronteira
                came_from[neighbor] = current_node  # Rastreia de onde viemos


    # Reporta os valores de desempenho finais se nenhum caminho for encontrado
    print(f"Total de nós visitados: {visited_nodes_count}")
    print(f"Tamanho máximo da fronteira: {max_frontier_size}")
    
    # Se o loop termina e não encontramos o objetivo
    return None  # Retorna None se nenhum caminho for encontrado

def breadth_first_search(graph):
    """
    Implementação da Busca em Largura (Breadth-First Search).
    
    :param graph: Instância da classe Graph.
    :return: Uma lista representando o caminho do nó inicial ao nó objetivo, se encontrado.
    """
    start = graph.edges['estado inicial']
    goal = graph.edges['estado final']

    # Inicializa a fronteira (queue) com o nó de início
    frontier = deque([start])
    
    # Dicionário para rastrear o caminho
    came_from = {}
    came_from[start] = None
    
    # Inicializa o contador de nós visitados e o tamanho máximo da fronteira
    visited_nodes_count = 0
    max_frontier_size = 0
    iteration = 1
    # Enquanto houver nós na fronteira para explorar
    while frontier:
        print(f"\nIteração {iteration}")
        iteration += 1
        # Atualiza o desempenho da fronteira
        max_frontier_size = desempenhoFronteira(frontier, max_frontier_size)
        
        # Remove o primeiro nó da fila (FIFO)
        current_node = frontier.popleft()
        
        # Atualiza o desempenho dos nós visitados
        visited_nodes_count = desempenhoNos(visited_nodes_count)
        
        # Se o nó atual é o objetivo, reconstruir o caminho e retornar
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            
            # Reporta os valores de desempenho finais
            print(f"\n\nTotal de nós visitados: {visited_nodes_count}")
            print(f"Tamanho máximo da fronteira: {max_frontier_size}")
            
            return path[::-1]  # Retorna o caminho do início ao objetivo
        
        # Explora os vizinhos do nó atual
        for neighbor, _ in graph.get_neighbors(current_node):
            if neighbor not in came_from:  # Se o vizinho ainda não foi explorado
                frontier.append(neighbor)  # Adiciona o vizinho à fila
                came_from[neighbor] = current_node  # Rastreia de onde viemos
    
    # Reporta os valores de desempenho finais se nenhum caminho for encontrado
    print(f"Total de nós visitados: {visited_nodes_count}")
    print(f"Tamanho máximo da fronteira: {max_frontier_size}")

    # Se o loop termina e não encontramos o objetivo
    return None  # Retorna None se nenhum caminho for encontrado

def uniform_cost_search(graph):
    """
    Implementação da Busca de Custo Uniforme (Uniform Cost Search).
    
    :param graph: Instância da classe Graph.
    :return: Uma lista representando o caminho do nó inicial ao nó objetivo, se encontrado.
    """
    start = graph.edges['estado inicial']
    goal = graph.edges['estado final']

    # Inicializa a fronteira (priority queue) com o nó de início
    frontier = [(0, start)]  # A fronteira é uma lista de tuplas (custo, nó)
    
    # Dicionário para rastrear o caminho
    came_from = {}
    came_from[start] = None
    
    # Dicionário para rastrear o menor custo até cada nó
    cost_so_far = {}
    cost_so_far[start] = 0
    
    # Inicializa o contador de nós visitados e o tamanho máximo da fronteira
    visited_nodes_count = 0
    max_frontier_size = 0
    iteration = 1
    # Enquanto houver nós na fronteira para explorar
    while frontier:
        print(f"\nIteração {iteration}")
        iteration += 1
        # Atualiza o desempenho da fronteira
        max_frontier_size = desempenhoFronteira(frontier, max_frontier_size)
        
        # Remove o nó com o menor custo total (usando uma priority queue)
        current_cost, current_node = heapq.heappop(frontier)
        
        # Atualiza o desempenho dos nós visitados
        visited_nodes_count = desempenhoNos(visited_nodes_count)
        
        # Se o nó atual é o objetivo, reconstruir o caminho e retornar
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            
            # Reporta os valores de desempenho finais
            print(f"\n\nTotal de nós visitados: {visited_nodes_count}")
            print(f"Tamanho máximo da fronteira: {max_frontier_size}")
            
            return path[::-1]  # Retorna o caminho do início ao objetivo
        
        # Explora os vizinhos do nó atual
        for neighbor, cost in graph.get_neighbors(current_node):
            new_cost = current_cost + cost  # Calcula o novo custo até o vizinho
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost  # Atualiza o menor custo para o vizinho
                heapq.heappush(frontier, (new_cost, neighbor))  # Adiciona o vizinho à fronteira com o novo custo
                came_from[neighbor] = current_node  # Rastreia de onde viemos
    
    # Reporta os valores de desempenho finais se nenhum caminho for encontrado
    print(f"Total de nós visitados: {visited_nodes_count}")
    print(f"Tamanho máximo da fronteira: {max_frontier_size}")

    # Se o loop termina e não encontramos o objetivo
    return None  # Retorna None se nenhum caminho for encontrado

def iterative_deepening_search(graph):
    """
    Implementação da Busca em Aprofundamento Iterativo (Iterative Deepening Search).
    
    :param graph: Instância da classe Graph.
    :return: Uma lista representando o caminho do nó inicial ao nó objetivo, se encontrado.
    """
    # Inicializa a profundidade máxima
    depth = 0
    visited_nodes_count = 0
    max_frontier_size = 0
    
    while True:
        # Executa a busca em profundidade limitada até a profundidade atual
        result, visited_nodes_count, max_frontier_size = depth_limited_search(graph, depth, visited_nodes_count, max_frontier_size)
        
        # Se o objetivo for encontrado, retorna o caminho
        if result is not None:
            # Reporta os valores de desempenho finais
            print(f"Total de nós visitados: {visited_nodes_count}")
            print(f"Tamanho máximo da fronteira: {max_frontier_size}")
            return result
        
        # Incrementa a profundidade para a próxima iteração
        depth += 1

def depth_limited_search(graph, limit, visited_nodes_count, max_frontier_size):
    """
    Implementação auxiliar de Busca em Profundidade Limitada (Depth-Limited Search).
    
    :param graph: Instância da classe Graph.
    :param limit: A profundidade máxima permitida para esta iteração.
    :param visited_nodes_count: Contador de nós visitados.
    :param max_frontier_size: O tamanho máximo da fronteira observado até agora.
    :return: Uma lista representando o caminho do nó inicial ao nó objetivo, se encontrado, ou None.
    """

    start = graph.edges['estado inicial']
    goal = graph.edges['estado final']
    
    # Função interna que realiza a busca em profundidade até o limite
    def recursive_dls(node, goal, limit, path, visited):
        nonlocal visited_nodes_count, max_frontier_size
        
        # Atualiza o desempenho da fronteira
        max_frontier_size = desempenhoFronteira(path, max_frontier_size)
        
        # Adiciona o nó atual ao caminho
        path.append(node)
        visited_nodes_count = desempenhoNos(visited_nodes_count)  # Atualiza o contador de nós visitados
        
        # Se o nó atual é o objetivo, retorna o caminho
        if node == goal:
            return path
        
        # Se atingimos o limite de profundidade, retorna None
        if limit <= 0:
            path.pop()
            return None
        
        # Marca o nó atual como visitado
        visited.add(node)
        
        # Explora os vizinhos do nó atual
        for neighbor, _ in graph.get_neighbors(node):
            if neighbor not in visited:
                result = recursive_dls(neighbor, goal, limit - 1, path, visited)
                if result is not None:
                    return result
        
        # Se nenhum caminho é encontrado, remove o nó do caminho
        path.pop()
        return None

    # Chama a função recursiva com os parâmetros iniciais
    result = recursive_dls(start, goal, limit, [], set())
    return result, visited_nodes_count, max_frontier_size

def depth_first_search_no_backtracking(graph):
    """
    Implementação da Busca em Profundidade sem Backtracking.
    
    :param graph: Instância da classe Graph.
    :return: Uma lista representando o caminho do nó inicial ao nó objetivo, se encontrado.
    """

    start = graph.edges['estado inicial']
    goal = graph.edges['estado final']

    # Inicializa o caminho com o nó de início
    path = [start]
    
    # Define o nó atual como o nó de início
    current_node = start

    # Inicializa o contador de nós visitados e o tamanho máximo da fronteira
    visited_nodes_count = 0
    max_frontier_size = 0
    iteration = 1
    # Enquanto houver estados a serem explorados
    while True:
        # Atualiza o desempenho da fronteira
        print(f"\nIteração {iteration}")
        iteration += 1
        max_frontier_size = desempenhoFronteira(path, max_frontier_size)
        
        # Atualiza o desempenho dos nós visitados
        visited_nodes_count = desempenhoNos(visited_nodes_count)
        
        # Se o nó atual é o objetivo, retorna o caminho
        if current_node == goal:
            # Reporta os valores de desempenho finais
            print(f"\n\nTotal de nós visitados: {visited_nodes_count}")
            print(f"Tamanho máximo da fronteira: {max_frontier_size}")
            return path
        
        # Obtém os vizinhos do nó atual
        neighbors = graph.get_neighbors(current_node)

        # Filtra os vizinhos já visitados
        unvisited_neighbors = [neighbor for neighbor, _ in neighbors if neighbor not in path]
        
        if unvisited_neighbors:
            # Se houver vizinhos não visitados, segue em frente
            next_node = unvisited_neighbors[0]
            path.append(next_node)
            current_node = next_node
        else:
            # Se não houver vizinhos não visitados, imprime que não encontrou solução
            print("Nenhuma solução encontrada a partir do estado atual.")
            
            # Reporta os valores de desempenho finais
            print(f"Total de nós visitados: {visited_nodes_count}")
            print(f"Tamanho máximo da fronteira: {max_frontier_size}")
            
            return None

def depth_first_search_with_backtracking(graph):
    """
    Implementação da Busca em Profundidade com Backtracking (DFS).
    
    :param graph: Instância da classe Graph.
    :return: Uma lista representando o caminho do nó inicial ao nó objetivo, se encontrado.
    """

    start = graph.edges['estado inicial']
    goal = graph.edges['estado final']
    
    # Inicializa o contador de nós visitados e o tamanho máximo da fronteira
    visited_nodes_count = 0
    max_frontier_size = 0

    # Função auxiliar recursiva para realizar a busca
    def dfs_recursive(current_node, goal, path, visited):
        nonlocal visited_nodes_count, max_frontier_size
        
        # Atualiza o desempenho da fronteira
        max_frontier_size = desempenhoFronteira(path, max_frontier_size)
        
        # Adiciona o nó atual ao caminho
        path.append(current_node)
        visited.add(current_node)
        
        # Atualiza o desempenho dos nós visitados
        visited_nodes_count = desempenhoNos(visited_nodes_count)
        
        # Se o nó atual é o objetivo, retorna o caminho
        if current_node == goal:
            # Reporta os valores de desempenho finais
            print(f"\n\nTotal de nós visitados: {visited_nodes_count}")
            print(f"Tamanho máximo da fronteira: {max_frontier_size}")
            return path
        
        # Explora os vizinhos do nó atual
        for neighbor, _ in graph.get_neighbors(current_node):
            if neighbor not in visited:  # Se o vizinho ainda não foi visitado
                result = dfs_recursive(neighbor, goal, path, visited)
                if result:  # Se encontramos o objetivo, retorna o caminho
                    return result
        
        # Se nenhum caminho é encontrado, remove o nó do caminho (backtracking)
        path.pop()
        return None
    
    # Inicializa os conjuntos de nós visitados e o caminho
    visited = set()
    path = []
    
    # Inicia a busca a partir do nó inicial
    result = dfs_recursive(start, goal, path, visited)
    
    # Se não encontrou solução, ainda reporta os valores de desempenho finais
    if result is None:
        print(f"Total de nós visitados: {visited_nodes_count}")
        print(f"Tamanho máximo da fronteira: {max_frontier_size}")
    
    return result

def a_star_search(graph, heuristic):
    """
    Implementação do Algoritmo A* com um dicionário de heurísticas.
    
    :param graph: Instância da classe Graph para o grafo principal.
    :param heuristic: Dicionário que contém os valores da heurística para cada nó.
    :return: Uma lista representando o caminho do nó inicial ao nó objetivo, se encontrado.
    """

    start = graph.edges['estado inicial']
    goal = graph.edges['estado final']

    # Inicializa a fronteira (priority queue) com o nó de início
    frontier = [(0, start)]  # A fronteira é uma lista de tuplas (custo total estimado, nó)
    
    # Dicionário para rastrear o caminho
    came_from = {}
    came_from[start] = None
    
    # Dicionário para rastrear o menor custo até cada nó
    cost_so_far = {}
    cost_so_far[start] = 0
    
    # Inicializa o contador de nós visitados e o tamanho máximo da fronteira
    visited_nodes_count = 0
    max_frontier_size = 0
    iteration = 1

    # Enquanto houver nós na fronteira para explorar
    while frontier:

        print(f"\nIteração {iteration}")
        iteration += 1

        # Remove o nó com o menor custo total estimado (f = g + h)
        current_cost, current_node = heapq.heappop(frontier)

        # Atualiza o desempenho dos nós visitados
        visited_nodes_count = desempenhoNos(visited_nodes_count)
        
        # Atualiza o desempenho da fronteira após a remoção do nó
        max_frontier_size = desempenhoFronteira(frontier, max_frontier_size)

        
        # Se o nó atual é o objetivo, reconstruir o caminho e retornar
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            
            # Reporta os valores de desempenho finais
            print(f"\n\nTotal de nós visitados: {visited_nodes_count}")
            print(f"Tamanho máximo da fronteira: {max_frontier_size}")
            
            return path[::-1]  # Retorna o caminho do início ao objetivo
        
        # Explora os vizinhos do nó atual
        for neighbor, cost in graph.get_neighbors(current_node):
            new_cost = cost_so_far[current_node] + cost  # g(n): custo atual acumulado até o vizinho
            
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                
                # Obtém a heurística h(n) para o vizinho do dicionário de heurísticas
                heuristic_value = heuristic.get(neighbor, float('inf'))  # Caso o nó não tenha heurística, considera infinito
                
                # Calcula f(n) = g(n) + h(n)
                priority = new_cost + heuristic_value
                heapq.heappush(frontier, (priority, neighbor))
                
                # Atualiza o caminho de onde viemos
                came_from[neighbor] = current_node
    
    # Reporta os valores de desempenho finais se nenhum caminho for encontrado
    print(f"Total de nós visitados: {visited_nodes_count}")
    print(f"Tamanho máximo da fronteira: {max_frontier_size}")

    # Se o loop termina e não encontramos o objetivo
    return None  # Retorna None se nenhum caminho for encontrado

def sma_star(graph, heuristic, memory_limit):
    """
    Implementação do Algoritmo SMA* (Simplified Memory-Bounded A*) com um dicionário de heurísticas.
    
    :param graph: Instância da classe Graph para o grafo principal.
    :param heuristic: Dicionário que contém os valores da heurística para cada nó.
    :param memory_limit: O número máximo de nós que podem ser mantidos na memória.
    :return: Uma lista representando o caminho do nó inicial ao nó objetivo, se encontrado.
    """

    start = graph.edges['estado inicial']
    goal = graph.edges['estado final']

    class Node:
        def __init__(self, state, parent=None, g=0, h=0):
            self.state = state
            self.parent = parent
            self.g = g  # Custo do caminho até este nó
            self.h = h  # Valor heurístico
            self.f = g + h  # Função de avaliação f(n) = g(n) + h(n)
            self.children = []
            self.fallback = float('inf')  # Melhor valor de f entre os filhos quando um nó é descartado

        def __lt__(self, other):
            return self.f < other.f

    def reconstruct_path(node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]

    # Inicializa a fronteira com o nó inicial
    start_node = Node(state=start, h=heuristic.get(start, float('inf')))
    frontier = [start_node]
    visited = {}
    
    # Inicializa o contador de nós visitados e o tamanho máximo da fronteira
    visited_nodes_count = 0
    max_frontier_size = 0
    iteration = 1

    while frontier:
        print(f"\nIteração {iteration}")
        iteration += 1
        
        # Ordena a fronteira para garantir que o nó com menor f seja processado primeiro
        heapq.heapify(frontier)
        
        # Remove o nó com o menor f(n)
        current_node = heapq.heappop(frontier)
        
        # Atualiza o desempenho dos nós visitados
        visited_nodes_count = desempenhoNos(visited_nodes_count)
        
        # Atualiza o desempenho da fronteira após a remoção do nó
        max_frontier_size = desempenhoFronteira(frontier, max_frontier_size)

        # Se o objetivo for encontrado, reconstrua o caminho e retorne
        if current_node.state == goal:
            # Reporta os valores de desempenho finais
            print(f"\n\nTotal de nós visitados: {visited_nodes_count}")
            print(f"Tamanho máximo da fronteira: {max_frontier_size}")
            
            return reconstruct_path(current_node)

        # Expande o nó atual
        neighbors = graph.get_neighbors(current_node.state)
        for neighbor, cost in neighbors:
            if neighbor in visited and visited[neighbor].g <= current_node.g + cost:
                continue  # Já visitado com um caminho mais curto

            h = heuristic.get(neighbor, float('inf'))
            child_node = Node(state=neighbor, parent=current_node, g=current_node.g + cost, h=h)
            current_node.children.append(child_node)

            # Adiciona o nó à fronteira
            heapq.heappush(frontier, child_node)
            visited[neighbor] = child_node

        # Se a memória estiver cheia, descarta o pior nó (com maior f) e atualiza o nó pai
        if len(frontier) > memory_limit:
            # Encontrar o nó com o maior f
            worst_node = max(frontier, key=lambda node: node.f)
            frontier.remove(worst_node)

            # Atualizar o fallback do pai
            if worst_node.parent:
                worst_node.parent.fallback = min(worst_node.parent.fallback, worst_node.f)
                if worst_node.parent in frontier:
                    heapq.heappush(frontier, worst_node.parent)

            # Reorganizar a fronteira
            heapq.heapify(frontier)

    # Reporta os valores de desempenho finais se a busca falhar
    print(f"Total de nós visitados: {visited_nodes_count}")
    print(f"Tamanho máximo da fronteira: {max_frontier_size}")

    # Se a busca falhar, retorna None
    return None

def ida_star(graph, heuristic):
    """
    Implementação do Algoritmo IDA* (Iterative Deepening A*) com um dicionário de heurísticas.
    
    :param graph: Instância da classe Graph para o grafo principal.
    :param heuristic: Dicionário que contém os valores da heurística para cada nó.
    :return: Uma lista representando o caminho do nó inicial ao nó objetivo, se encontrado.
    """

    start = graph.edges['estado inicial']
    goal = graph.edges['estado final']
    
    visited_nodes_count = 0
    max_frontier_size = 0
    iteration_count = 0  # Contador de iterações

    def search(node, g, threshold, path):
        """
        Função recursiva auxiliar para realizar a busca com limitação de profundidade adaptativa.
        
        :param node: O nó atual.
        :param g: O custo acumulado até o nó atual.
        :param threshold: O valor limite para f(n) na iteração atual.
        :param path: A lista que mantém o caminho atual.
        :return: O novo limite de f, ou o caminho se o objetivo for encontrado.
        """
        nonlocal visited_nodes_count, max_frontier_size, iteration_count
        
        # Mostrar a iteração atual antes de qualquer atualização de desempenho
        print(f"\nIteração {iteration_count}")
        
        f = g + heuristic.get(node, float('inf'))  # Calcula f(n) = g(n) + h(n)
        
        # Atualiza o contador de nós visitados (com print dentro da função)
        if node not in path:
            visited_nodes_count = desempenhoNos(visited_nodes_count)

        if f > threshold:
            return f, None  # Retorna o novo limite

        if node == goal:
            return f, path + [node]  # Objetivo encontrado, retorna o caminho

        min_threshold = float('inf')
        for neighbor, cost in graph.get_neighbors(node):
            if neighbor not in path:  # Evita ciclos
                new_path = path + [node]
                
                # Atualiza o desempenho da fronteira (com print dentro da função)
                max_frontier_size = desempenhoFronteira(new_path, max_frontier_size)

                new_g = g + cost
                t, result = search(neighbor, new_g, threshold, new_path)
                if result is not None:
                    return t, result  # Caminho encontrado
                min_threshold = min(min_threshold, t)

        return min_threshold, None

    # Inicializa o limiar como a heurística do nó inicial
    threshold = heuristic.get(start, float('inf'))

    # Loop iterativo, aumentando o limiar até encontrar a solução
    while True:
        print(f"\nIniciando iteração com limiar {threshold}")
        iteration_count += 1  # Incrementa o contador de iterações
        t, result = search(start, 0, threshold, [])
        
        if result is not None:
            # Apenas imprime as métricas finais, já que as intermediárias foram mostradas pelas funções de desempenho
            print(f"\nTotal de nós visitados: {visited_nodes_count}")
            print(f"Tamanho máximo da fronteira observado: {max_frontier_size}")
            return result  # Caminho encontrado
        
        if t == float('inf'):
            # Apenas imprime as métricas finais, já que as intermediárias foram mostradas pelas funções de desempenho
            print(f"Total de nós visitados: {visited_nodes_count}")
            print(f"Tamanho máximo da fronteira observado: {max_frontier_size}")
            return None  # Não há solução
        
        threshold = t  # Atualiza o limiar para a próxima iteração

def format_path(path):
    formated_path = ""
    for index, node in enumerate(path):
        formated_path += node 
        if index != (len(path) - 1):
            formated_path += " -> "

    return formated_path
        
def path_cost(graph, path):
    """
    Calcula o custo total de um caminho em um grafo.

    :param graph: Instância da classe Graph que contém as arestas e os custos entre os nós.
    :param path: Lista de nós que representam o caminho encontrado por um algoritmo de busca.
    :return: O custo total do caminho.
    """
    # Verifica se o caminho está vazio ou possui um único nó
    if not path or len(path) == 1:
        return 0  # Custo é zero, pois não há transições

    total_cost = 0  # Inicializa o custo total do caminho

    # Itera sobre o caminho, pegando o par de nós consecutivos
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        
        # Obtém os vizinhos do nó atual e encontra o custo para o próximo nó
        neighbors = graph.get_neighbors(from_node)
        
        # Busca o custo da aresta que leva ao próximo nó
        for neighbor, cost in neighbors:
            if neighbor == to_node:
                total_cost += cost
                break
        else:
            # Se não houver aresta direta entre os nós consecutivos, há um problema
            print(f"Erro: Não foi encontrada uma aresta entre {from_node} e {to_node}.")
            return None

    return total_cost

def iterate_through_samples():
    return

def main():
    graph = Graph()

    fileName = input('Digite o nome do arquivo: ')
    read_graph_from_file(fileName, graph)

    #Apresenta o grafo formatado
    print(format_graph(graph))
    #print(format_graph(graph.edges))
    
    print('Qual algoritmo deseja executar?')
    while True:
        opcao = input('''Ótimos:
    1 - SMA* (Selecionado como melhor baseado nas medidas de desempenho)
    2 - A*
    3 - IDA*
    4 - Custo uniforme

    Não ótimos:
    5 - Profundidade sem backtracking (Selecionado como pior baseado nas medidas de desempenho)
    6 - Busca em largura
    7 - Aprofundamento interativo
    8 - Busca gulosa
    9 - Profundidade com backtracking

    0 - Encerrar o programa

    Digite a opção desejada (0 a 9): 
    ''')

        if opcao == '0':
            print("Encerrando o programa...")
            break
        elif opcao == '1':
            print("\n\nBusca SMA*")
            path = sma_star(graph, graph.heuristic, 5)
            print(f"Caminho encontrado: {format_path(path)}\nCusto do caminho: {path_cost(graph, path)}\n\n")
        elif opcao == '2':
            print("\n\nBusca A*")
            path = a_star_search(graph, graph.heuristic)
            print(f"Caminho encontrado: {format_path(path)}\nCusto do caminho: {path_cost(graph, path)}\n\n")
        elif opcao == '3':
            print("\n\nBusca IDA*")
            path = ida_star(graph, graph.heuristic)
            print(f"Caminho encontrado: {format_path(path)}\nCusto do caminho: {path_cost(graph, path)}\n\n")
        elif opcao == '4':
            print("\n\nBusca de Custo Uniforme:")
            path = uniform_cost_search(graph)
            print(f"Caminho encontrado: {format_path(path)}\nCusto do caminho: {path_cost(graph, path)}\n\n")
        elif opcao == '5':
            print("\n\nBusca em profundidade sem backtracking")
            path = depth_first_search_no_backtracking(graph)
            print(f"Caminho encontrado: {format_path(path)}\nCusto do caminho: {path_cost(graph, path)}\n\n")
        elif opcao == '6':
            print("\n\nBusca em Largura:")
            path = breadth_first_search(graph)
            print(f"Caminho encontrado: {format_path(path)}\nCusto do caminho: {path_cost(graph, path)}\n\n")
        elif opcao == '7':
            print("\n\nBusca em Aprofundamento Iterativo:")
            path = iterative_deepening_search(graph)
            print(f"Caminho encontrado: {format_path(path)}\nCusto do caminho: {path_cost(graph, path)}\n\n")
        elif opcao == '8':
            print("\n\nBusca Gulosa:")
            path = greedy_search(graph, graph.heuristic)
            print(f"Caminho encontrado: {format_path(path)}\nCusto do caminho: {path_cost(graph, path)}\n\n")
        elif opcao == '9':
            print("\n\nBusca em profundidade com backtracking")
            path = depth_first_search_with_backtracking(graph)
            print(f"Caminho encontrado: {format_path(path)}\nCusto do caminho: {path_cost(graph, path)}\n\n")
        else:
            print("Opção inválida! Por favor, escolha um número entre 0 e 9.")

if __name__ == "__main__":
    main()
