
import numpy as np  # 用于科学计算
import matplotlib.pyplot as plt  # 用于图形显示
import matplotlib.cm as cm  # 用于颜色映射
import random  # 用于生成随机数
import math  # 用于数学运算
import argparse  # 用于解析命令行参数



class Node:
    '''
    表示网络中的单个节点。
    '''
    def __init__(self, value, number, connections=None):
        '''
        初始化一个节点。
        :param value: 节点的数值
        :param number: 节点的索引
        :param connections: 节点与其他节点的连接状态列表，默认为None
        '''
        self.index = number  # 节点索引
        self.connections = connections if connections is not None else []  # 初始化连接状态
        self.value = value  # 节点的数值

class Network:
    '''
    表示节点的网络。
    '''

    def __init__(self, nodes=None):
        '''
        初始化网络。
        :param nodes: 网络中的节点列表，默认为None
        '''
        self.nodes = nodes if nodes is not None else []  # 初始化节点列表

    def get_mean_degree(self):
        '''
        计算网络的平均度数。
        '''
        total_degree = sum(sum(node.connections) for node in self.nodes)
        return total_degree / len(self.nodes)

    def get_mean_clustering(self):
        '''
        计算网络的平均聚类系数。
        '''
        total_cc = 0
        for node in self.nodes:
            neighbours = [self.nodes[i] for i, conn in enumerate(node.connections) if conn]
            num_neighbours = len(neighbours)
            if num_neighbours <= 2:
                continue
            possible_triangles = num_neighbours * (num_neighbours - 1) / 2
            actual_triangles = 0
            for i in range(1, num_neighbours):
                for j in range(i + 1, num_neighbours):
                    if node.connections[neighbours[i].index] and node.connections[neighbours[j].index]:
                        actual_triangles += 1
            cc = actual_triangles / possible_triangles if possible_triangles != 0 else 0
            total_cc += cc
        return total_cc / len(self.nodes)

    def get_mean_path_length(self):
        '''
        计算网络的平均路径长度。
        '''
        total_path_length = 0
        for node in self.nodes:
            distances = self.bfs(node)
            total_path_length += sum(distances.values())
        return total_path_length / (len(self.nodes) * (len(self.nodes) - 1))

    def bfs(self, start_node):
        '''
        从一个起始节点开始执行广度优先搜索。
        :param start_node: 起始节点
        '''
        distances = {node.index: float('inf') for node in self.nodes}
        distances[start_node.index] = 0
        queue = [start_node]
        while queue:
            current_node = queue.pop(0)
            for neighbour_index, conn in enumerate(current_node.connections):
                if conn and distances[neighbour_index] == float('inf'):
                    distances[neighbour_index] = distances[current_node.index] + 1
                    queue.append(self.nodes[neighbour_index])
            for node in self.nodes:
                if node.connections[current_node.index] and distances[node.index] == float('inf'):
                    distances[node.index] = distances[current_node.index] + 1
                    queue.append(node)
        return distances

    def make_random_network(self, N, connection_probability):
        '''
        创建一个规模为N的随机网络。
        每个节点之间按概率p进行连接。
        '''
        self.nodes = []  # 初始化节点列表
        for node_number in range(N):  # 对于每个节点编号
            value = np.random.random()  # 生成一个[0,1)之间的随机浮点数，作为节点的值
            connections = [0 for _ in range(N)]  # 初始化节点的连接列表，所有元素初始值为0
            self.nodes.append(Node(value, node_number, connections))  # 创建节点并加入网络
        
        # 保证每个节点至少有一个连接
        for (index, node) in enumerate(self.nodes):
            other_node_index = random.choice([i for i in range(N) if i != index])
            node.connections[other_node_index] = 1  # 随机选择一个其他节点进行连接
            self.nodes[other_node_index].connections[index] = 1  # 双向连接

        # 根据给定的连接概率p添加额外的连接
        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:  # 随机决定是否连接
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1  # 双向连接

        # 计算网络的一些统计特性
        mean_degree = self.get_mean_degree()  # 平均度数
        mean_clustering_coefficient = self.get_mean_clustering()  # 平均聚类系数
        mean_path_length = self.get_mean_path_length()  # 平均路径长度

        # 打印计算结果
        print("Mean degree:", mean_degree)
        print("Mean clustering coefficient:", mean_clustering_coefficient)
        print("Mean path length:", mean_path_length)

        # 生成和绘制节点位置及其连接
        node_coordinates = {node: (np.random.uniform(0, N), np.random.uniform(0, N)) for node in range(N)}
        plt.figure()
        for node in range(N):
            x1, y1 = node_coordinates[node]
            plt.plot(x1, y1, 'o', color='black')  # 绘制节点
            for neighbour_index, conn in enumerate(self.nodes[node].connections):
                if conn:
                    x2, y2 = node_coordinates[neighbour_index]
                    plt.plot([x1, x2], [y1, y2], '-', color='black')  # 绘制连接线
        plt.title("Random Network")
        plt.show()  # 显示图形

    def make_ring_network(self, N, neighbour_range=1):
        '''
        创建一个环形网络，每个节点连接其最近的几个邻居。
        :param N: 网络中节点的数量
        :param neighbour_range: 邻居范围
        '''
        self.nodes = [Node(0, x) for x in range(N)]  # 创建N个节点，初始化它们的连接
        for node in self.nodes:
            node.connections = [0 for _ in range(N)]  # 初始化连接列表
            for neighbour_index in range(-neighbour_range, neighbour_range + 1):  # 对于每个邻居范围
                if neighbour_index != 0:
                    node.connections[(node.index + neighbour_index) % N] = 1  # 设置周期边界条件

    def make_small_world_network(self, N, re_wire_prob=0.2):
        '''
        创建一个小世界网络。
        :param N: 网络中节点的数量
        :param re_wire_prob: 重连概率
        '''
        self.make_ring_network(N, 2)  # 先创建一个环形网络，每个节点连接最近的两个邻居
        edges = []
        # 收集所有已经存在的边
        for node in self.nodes:
            for neighbour_pos in range(node.index, N):
                if node.connections[neighbour_pos] == 1:
                    edges.append((node.index, neighbour_pos))

        # 根据重连概率重新配置连接
        for edge in edges:
            if np.random.random() <= re_wire_prob:
                start_node, end_node = self.nodes[edge[0]], self.nodes[edge[1]]
                combined_connections = [a or b for a, b in zip(start_node.connections, end_node.connections)]
                possible_rewires = [i for i in range(N) if combined_connections[i] == 0]
                try:
                    chosen_rewire = possible_rewires[np.random.randint(0, len(possible_rewires) + 1)]
                    new_end_node = self.nodes[chosen_rewire]
                    start_node.connections[edge[1]] = 0
                    end_node.connections[edge[0]] = 0
                    start_node.connections[chosen_rewire] = 1
                    new_end_node.connections[edge[0]] = 1
                except:
                    pass  # 如果没有找到可重连的节点则跳过

    def plot(self):
        # 创建一个新的图形
        fig = plt.figure()
        # 添加一个子图，'111'表示图形网格为1x1网格的第1个
        ax = fig.add_subplot(111)
        # 关闭坐标轴
        ax.set_axis_off()

        # 计算网络中的节点总数
        num_nodes = len(self.nodes)
        # 设定网络的半径，基于节点数量扩展
        network_radius = num_nodes * 10
        # 设置x轴和y轴的限制，保证所有节点能在图内显示
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        # 遍历每个节点以绘制它们
        for (i, node) in enumerate(self.nodes):
            # 计算节点在圆上的角度
            node_angle = i * 2 * np.pi / num_nodes
            # 计算节点的x和y坐标
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            # 创建一个表示节点的圆形，并设置颜色
            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            # 将圆形添加到图中
            ax.add_patch(circle)

            # 遍历邻居节点，并绘制连接线
            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    # 计算邻居节点的角度
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    # 计算邻居节点的x和y坐标
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    # 在两个节点之间画线
                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')


# 创建一个 Network 实例
network = Network()

# 创建一个规模为 N 的随机网络
N = 20  # 网络中节点的数量
connection_probability = 0.2  # 节点之间的连接概率
network.make_random_network(N, connection_probability)

# 注意：make_random_network 方法内部会打印网络的平均度数、平均聚类系数和平均路径长度等统计特性

# 调用 plot 方法绘制网络
network.plot()

# 从网络中选择索引为10的节点
selected_node = network.nodes[10]

# 打印该节点的数值和连接状态
print("Selected Node Value:", selected_node.value)
print("Selected Node Connections:", selected_node.connections)


# 计算格点模型的能量
def calculate_agreement_grid(x, y, grid, H):
    total_agreement = 0
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    for nx, ny in neighbors:
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            total_agreement += grid[x, y] * grid[nx, ny]
    total_agreement += H * grid[x, y]
    return total_agreement

# 计算网络模型的能量
def calculate_agreement_network(node_index, network, H):
    node = network.nodes[node_index]
    total_agreement = 0
    for i, connected in enumerate(node.connections):
        if connected:
            total_agreement += node.value * network.nodes[i].value
    total_agreement += H * node.value
    return total_agreement

# 统一的能量计算函数，根据输入类型调用相应的计算函数
def calculate_agreement(*args):
    if isinstance(args[0], int):  # 如果输入是 int 类型，则假设为格点模型
        return calculate_agreement_grid(*args)
    else:  # 否则假设为网络模型
        return calculate_agreement_network(*args)
def calculate_agreement(x, y, grid,H):
    total_agreement = 0
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    for nx, ny in neighbors:
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            total_agreement += grid[x, y] * grid[nx, ny]
    total_agreement += H * grid[x, y]
    return total_agreement


def ising_step(grid, alpha, H):
    x, y = np.random.randint(0, len(grid)), np.random.randint(0, len(grid[0]))
    Di = calculate_agreement(x, y, grid, H)

    if Di < 0 or np.random.random() < np.exp(-Di / alpha):
        grid[x, y] *= -1



def plot_ising(im, population):
    # Update the image data
    new_im = np.array([[-1 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    # Pause to update the image
    plt.pause(0.1)

def test_ising():

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    H_test = 0
    assert(calculate_agreement(1,1, population, H_test)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(1,1, population, H_test)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(1,1, population, H_test)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(1,1, population, H_test)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(1,1, population, H_test)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(1,1, population, H_test)==4), "Test 6"

    H_test = 1

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(1,1, population, H_test)== 3), "Test 7"
    assert(calculate_agreement(1,1, population, -1)==5), "Test 8"
    assert (calculate_agreement(1,1, population, 10) == -6), "Test 9"
    assert (calculate_agreement(1,1, population, -10) == 14), "Test 10"

    print("Tests passed")






def ising_main(population, alpha,H):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    for frame in range(100):
        for step in range(1000):
            ising_step(population, alpha, H)
        plot_ising(im, population)
        plt.pause(0.001)
    plt.ioff()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Run the ising model simulations using settings.')
    parser.add_argument('-ising_model', action='store_true', help='Run the ising model simulation using default settings.')
    parser.add_argument('-external', type=float, default=0.0, help='Set the strength of the external influence on the model.')
    parser.add_argument('-alpha', type=float, default=1.0, help='Set the alpha value used in the agreement calculation.')
    parser.add_argument('-test_ising', action='store_true', help='Run the ising model test functions to ensure integrity.')

    # Process the provided data
    args = parser.parse_args()

    if args.alpha <= 0:
        parser.error("The alpha parameter must be greater than 0.")

    if args.test_ising:
        test_ising()
    elif args.ising_model:
        population = np.random.choice([-1, 1], size=(100, 100))
        ising_main(population, args.alpha, args.external)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()


    # return np.random * population





'''
在终端运行

cd "/Users/johnwong/Documents/00 U Course/2023-2024 本地/FCP Summative Assessment"
#或者
python3 assignment.py -ising_model
python3 assignment.py -ising_model -external -0.1
python3 assignment.py -ising_model -alpha 10
python3 assignment.py -test_ising



'''

