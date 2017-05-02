import sys
import os

#sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../../pychain/src')))


import pickle
import time
from datetime import datetime
import matplotlib.pyplot as plt
from lstm_node import *
from layer import *
from recurrent_graph import *
from optimization_algorithm import *
from midi_conversion import *


# Read file
path = 'midi_datasets/mz_311_1_format0.mid'
midi = MidiFile(path)
matrix_midi = midiToMatrix(midi)


num_lstms = 4
dim_s = 512#253#128
learning_rate = 2e-3
len_seq = 12#9#2#10
nb_seq_per_batch = 50#3#50
hidden_shapes = [(nb_seq_per_batch, dim_s), (nb_seq_per_batch, dim_s)] * num_lstms


def learn_midi(layer):
    # Create the graph
    graph = RecurrentGraph(layer, len_seq - 1, hidden_shapes)
    # Optimization algorithm
    #algo = GradientDescent(graph.get_learnable_nodes(), learning_rate)
    algo = RMSProp(graph.get_learnable_nodes(), learning_rate, 0.95)
    # Learn
    i_pass = 1
    #save_layer(layer, 0)
    i_batch = 1
    while True:
        len_batch = len_seq*nb_seq_per_batch
        nb_batches = int(matrix_midi.shape[1] / len_batch)
        for i in range(nb_batches):
            t_start = time.time()
            # Take a new batch
            batch = matrix_midi[:,i*len_batch:(i+1)*len_batch]
            data = np.zeros((len_seq, nb_seq_per_batch, matrix_midi.shape[0]))
            for i_seq in range(nb_seq_per_batch):
                for index, note in enumerate(batch[:,i_seq * len_seq:(i_seq + 1) * len_seq].T):
                    data[index, i_seq] = note
            # Propagate and backpropagate the batch
            graph.propagate(data[:-1])
            cost = graph.backpropagate(data[1:]) / len_seq / nb_seq_per_batch
            # Get gradient and params norm
            lstm_node = layer.learnable_nodes[1]
            grad_norm = 0
            param_norm = 0
            for w in lstm_node.learnable_nodes:
                grad_norm += ((w.acc_dJdw/nb_seq_per_batch)**2).sum()
                param_norm += (w.w**2).sum()
            # Desend gradient
            algo.optimize(nb_seq_per_batch)
            # Print info
            print('pass: ' + str(i_pass) + ', batch: ' + str(i+1) + '/' + str(nb_batches) + \
                ', cost: ' + str(cost) + ', time: ' + str(time.time() - t_start)  + \
                ', grad/param norm: ' + str(np.sqrt(grad_norm/param_norm)))
            # Save
            if i_batch % 200 == 0:
                save_layer(layer, i_batch)
            i_batch += 1
        i_pass += 1

def create_layer():
    # Input
    x = InputNode()
    hidden_inputs = []
    hidden_outputs = []
    lstms = []
    # LSTMs
    parent = x
    for i in range(num_lstms):
        h_in = InputNode()
        s_in = InputNode()
        dim_x = dim_s
        if i == 0:
            dim_x = matrix_midi.shape[0]
        lstm = LSTMWFGNode(dim_x, dim_s, [parent, h_in, s_in])
        h_out = IdentityNode([(lstm, 0)])
        s_out = IdentityNode([(lstm, 1)])
        parent = h_out
        # Add to containers
        hidden_inputs += [h_in, s_in]
        hidden_outputs += [h_out, s_out]
        lstms.append(lstm)
    # Sigmoid
    w = LearnableNode(0.1 * np.random.randn(dim_s, matrix_midi.shape[0]))
    mult = MultiplicationNode([parent, w])
    out = SigmoidNode([mult])
    #out = TanhNode([mult])

    # Cost
    y = InputNode()
    cost = SigmoidCrossEntropyNode([y, out])
    #e = SubstractionNode([y, out])
    #cost = Norm2Node([e])

    nodes = hidden_inputs + hidden_outputs + lstms + [x, w, mult, out, y, cost]
    return Layer(nodes, [x], [out], hidden_inputs, hidden_outputs, [y], cost, [w] + lstms)


def save_layer(layer, i_batch):
    path = 'models/models_mz311_midi/' + str(datetime.now().strftime("%d-%m-%Y %Hh%Mmin%Ss")) + '_b-' +  str(i_batch) + '.pickle'
    pickle.dump(layer, open(path, 'wb'))

if __name__ == '__main__':
    layer = create_layer()
    learn_midi(layer)
