import torch
import os
from itertools import product
from graph_generation import *
from graph_model import *
from brute_force import *
from experiment import *

device = torch.device('cuda')


if not os.path.exists("./model"):
    os.makedirs("./model")
if not os.path.exists("./graph"):
    os.makedirs("./graph")
if not os.path.exists("./img"):
    os.makedirs("./img")

# def                 graph_generator(tree_or_graph, output_file,              in_features, num_graphs, num_nodes, randint_1, randint_2):
graphs, in_features = graph_generator("graph", 'random_undirected_graphs.pkl', 1,          10,         50,        499,         510)

n_heads_list = [8]
hidden_features_list = [64]#, 32, 128, 256]
out_features_list = [8]
n_epochs_list = [200]
lr_list = [0.0005]#, 0.05, 0.0005, 0.00005]
dropout_list = [0.1]#, 0.3, 0.5, 0.7]


params = list(product(n_heads_list, hidden_features_list, out_features_list, n_epochs_list, lr_list, dropout_list))
params_names = ['n_heads', 'hidden_features', 'out_features', 'n_epochs', 'lr', 'dropout']

for ex_num, param in enumerate(params):
    print(f'Experiment {ex_num}')
    print(param)
    param_dict = {}
    for idx, param_name in enumerate(params_names):
        param_dict[param_name] = param[idx]
    gat_model = run_experiment(in_features, graphs, param_dict)
    #print()

torch.save(gat_model, './model/gat_model1.pt')
model = torch.load('./model/gat_model1.pt')

#테스트용 그래프 생성
# def                 graph_generator(tree_or_graph, output_file,                  in_features, num_graphs, num_nodes, randint_1, randint_2):
trees, in_features = graph_generator("graph", 'random_undirected_graphs_test.pkl', 1,           2,          100,        99,         120)

bf_reward_list_gat = []
bf_path_list_gat = []
highest_reward_values_gat = [[] for _ in range(len(trees))]
for graph_idx, (x, adj_tensor, j, adj_matrix_original, _) in enumerate(trees):
    x = x.cuda()
    adj_tensor = adj_tensor.cuda()
    
    num_nodes = x.size(0)

    # epoch_segment = min(epoch // (n_epochs // num_nodes), num_nodes - 1)
    epoch_segment = 1
    visited_nodes = [epoch_segment]

    #그래디언트 초기화 
    output = model(x, adj_tensor)
    #print(output)
    #optimizer.zero_grad()

    # Decoder iteration
    visited_nodes, possibilities, rewards, previous_nodes = decode_all(model, output, x, adj_matrix_original, visited_nodes)
    node_weights = x
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype = np.float32)

    for i in range(len(visited_nodes)-1):
        start_node = previous_nodes[i]
        end_node = visited_nodes[i+1]
        adjacency_matrix[start_node][end_node] = 1
        adjacency_matrix[end_node][start_node] = 1
    adjacency_matrix = torch.tensor(adjacency_matrix)
    
    highest_reward_value, path_with_highest_reward = highest_rewards_for_end_nodes(epoch_segment, x, adjacency_matrix)
    #print("highest reward: " + str(highest_reward_value))
    bf_path_list_gat.append(path_with_highest_reward)
    # print(f'bf_path_list : {bf_path_list}')
    bf_reward_list_gat.append(highest_reward_value)

    highest_reward_values_gat[graph_idx].append(highest_reward_value)
    
    print(f'Graph {graph_idx}')
    for i in range(0, len(highest_reward_value)):
        print(f'end node : {i}')
        print("Highest Reward: " + str(highest_reward_value[i]) + " Corresponding Path: " + str(path_with_highest_reward[i]))
    print("visited_nodes", visited_nodes)
    print("previous_nodes", previous_nodes)
    #print("adjacency_matrix", adjacency_matrix)
    # print("adj_matrix_original", adj_matrix_original)
    print()

G = nx.Graph(trees[1][3].numpy())
# print("노드 수:", G.number_of_nodes())
# print("에지 수:", G.number_of_edges())

pos = nx.spring_layout(G)
plt.title("Original Graph")
nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
plt.show()
plt.savefig("./img/original.jpg")
plt.close()

G = nx.Graph(adjacency_matrix.numpy())
# print("노드 수:", G.number_of_nodes())
# print("에지 수:", G.number_of_edges())

pos = nx.spring_layout(G)
plt.title("Reconstructed Graph")
nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
plt.show()
plt.savefig("./img/recon.jpg")
plt.close()

bf_reward_list = []
bf_path_list = []
# print(f'bf_path_list: {bf_path_list}')
print("Calculating Brute Forcing reward...")

for graph_idx, (x, adj_tensor, j, adj_matrix_original, _) in enumerate(trees):
    start_node = 1 
    highest_reward_value, path_with_highest_reward = highest_rewards_for_end_nodes(start_node, x, adj_matrix_original)
    #print("highest reward: " + str(highest_reward_value))
    bf_path_list.append(path_with_highest_reward)
    # print(f'bf_path_list : {bf_path_list}')
    bf_reward_list.append(highest_reward_value)

    print(f'graph {graph_idx}')
    # print(path_with_highest_reward)
    # print(highest_reward_value)
    # print(bf_path_list)
    for i in range(0, len(highest_reward_value)):
        print(f'end node : {i}')
        print("Highest Reward: " + str(highest_reward_value[i]) + " Corresponding Path: " + str(path_with_highest_reward[i]))
    

 
# print(f'bf_path_list[graph_idx] : {bf_path_list[0]}')

# fig, axes = plt.subplots(len(graphs), 1, figsize=(8, 6*len(graphs)))0
plt.figure(figsize = (10, len(trees)*7))
for i in range(0, len(trees)):
    
    plt.subplot(len(trees)+1, 1, i+1)
    plt.title("Graph " + str(i+1) + " Comparison Graph")
    plt.xlabel("End Node")
    plt.ylabel("Attack Path Score")
    x_values = range(len(bf_reward_list[i]))
    y_values_bf = bf_reward_list[i]
    y_values_gat = bf_reward_list_gat[i]
    #y_values = bf_reward_list_gat[i]/bf_reward_list[i]
    # plt.scatter(x_values, y_values, c = 'r', label = "Bruete Force / GAT", marker = 'o')
    plt.scatter(x_values, y_values_bf, c='r', label="Brute Force", marker='o')
    plt.scatter(x_values, y_values_gat, c='b', label="GAT", marker='x')
    # plt.plot(range(0,len(bf_reward_list[i])), bf_reward_list[i],'r',label = "Brute Force" )
    # plt.plot(range(0,len(bf_reward_list_gat[i])), bf_reward_list_gat[i],'b', label = "GAT")
    plt.legend(loc = 'upper right')
plt.savefig("./img/comparison.jpg")