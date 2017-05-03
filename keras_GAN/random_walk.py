import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
You can use the adjacency matrix.
Then you can normalise it so that the sum of rows equals 1 and each row is the
 probability distribution of the node jumping to another node.
You can also have a jump probability if the walker jumps to a random node.
'''


class RandomWalk:

    def __init__(self,node_number = 20,probability_nodes = 0.3):
        self.G =  nx.gnp_random_graph(node_number, probability_nodes,directed=True)

    def graph_summary(self):
        print(self.G.number_of_edges())
        print(self.G.edges())
        print(self.G.out_degree())

    def trans_prob(self):
        i=0
        for (node,od) in self.G.out_degree().items():
            N = 1.0/od
            #print(random.choice(G.edges()[i:i+od]))
            i+=od

    # j is the starting point in the graph
    def random_walk(self,steps=5000,j=0,destination=5,restart_p = 0.9,orig=False):

        adj_mat = nx.adjacency_matrix(self.G).toarray().astype('float32')
        if orig:
            self.orig_mat = np.array(adj_mat)
            return(self.orig_mat)

        # j is the starting point, in this case, node 0
        num_steps = []
        ni = 0
        for steps in range(steps):
            # this calculates the number of steps until getting to destination!
            if steps > 0 and j == destination:
                if num_steps < 1:
                    num_steps.append(steps)
                    ni+=1
                else:
                    num_steps.append(steps)
                    num_steps[ni] = steps - num_steps[ni-1]

            # here we loop through each column in the adjacency matrix
            for i in range(adj_mat.shape[1]):

                # this ensures all transition probabilities sum to 1
                if(np.sum(adj_mat[j]) > 0.0):
                    # adj_mat[i] gives ith row
                    adj_mat[j] = adj_mat[j] / np.sum(adj_mat[j])

            # after normalization, if a random probability p is higher than 0.9,
            # we jump from one node to teleport to another.
            if random.random() > restart_p:
                # restart
                teleport = np.random.randint(0, adj_mat.shape[1])
                # we now teleport to the jth node
                j = teleport
                #restart_p = 1-restart_p

            # else we transition to a node with a certain probability on an outgoing link
            else:
                outlinks = [i for i, e in enumerate(adj_mat[j]) if e != 0.0]
                current_node = random.choice(outlinks)
                adj_mat[j][current_node] += np.mean(adj_mat[j])
                # move to node j in the adjanceny matrix
                j = current_node
                # choose next node

        mean_num_steps = np.mean([x - num_steps[i - 1] for i, x in enumerate(num_steps)][1:])
        print("It took an average of " + str(mean_num_steps) + " steps to return to the starting node")
        self.random_mat = np.array(adj_mat)
        return (self.random_mat)

    def visualize_walk(self):
        if self.orig_mat.any():
            print("Original")
            plt.figure()
            plt.imshow(self.orig_mat, cmap='hot', interpolation='nearest')
        plt.show()

        if self.random_mat.any():
            bounds = [0., 0.5, 15.0]
            print("New")

            plt.figure()
            cmap = mpl.colors.ListedColormap(['r', 'k'])
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            plt.imshow(orig_mat, interpolation='none', cmap=cmap)
        plt.show()


rw = RandomWalk()
orig_mat = rw.random_walk(orig=True)
random_mat = rw.random_walk()
print (orig_mat)
print (random_mat)
rw.visualize_walk()


