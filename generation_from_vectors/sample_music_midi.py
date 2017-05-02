import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))
import pickle
import matplotlib.pyplot as plt
from lstm_node import *
from layer import *
from recurrent_graph import *
from midi_conversion import *


# Read file
path = 'music/mz_311_1_format0.mid'
midi = MidiFile(path)
matrix_midi = midiToMatrix(midi)


num_lstms = 4
dim_s = 512
len_seq = 1000
hidden_shapes = [(1, dim_s), (1, dim_s)] * num_lstms

def note_to_vector(notes):
    #notes is a list which contains the note and its velocity
    #vector = np.zeros((1, matrix_midi.shape[0]))*2/127 -1 #between -1 and 1 for a final tanh node
    vector = np.zeros((1, matrix_midi.shape[0])) #between 0 and 1 for a final sigmoid node
    for note in notes:
        #vector[0, note[0]] = note[1]*2/127 - 1 #tanh
        vector[0, note[0]] = note[1] / 127 #sigmoid
    return vector

def argmax(output):
    chosen_note = np.zeros((1, matrix_midi.shape[0]))
    chosen_note[0,np.argmax(output)] = 1
    return chosen_note

def identity(output):
    vector = np.zeros((1, matrix_midi.shape[0]))
    for i in range(matrix_midi.shape[0]):
        vector[0,i] = output[0,i]*2.5
        #vector[0,i] = 1 / (1 + np.exp(-(output[0,i]*2-0.23)*20))
        #vector[0,i] = np.tanh(output[0,i]*2)
        #vector[0, i] = (output[0, i]*2) * (output[0,i]*2)
    return vector

def stochastic(output):
    #vector = -1*np.ones((1, matrix_midi.shape[0])) #tanh
    vector = np.zeros((1, matrix_midi.shape[0])) #sigmoid
    out = output
    for i in range (output.shape[1]):
        #out[0,i] = (output[0,i]+1)*0.5 #tanh
        out[0, i] = (output[0, i])  #sigmoid
    sum = np.sum(out)
    vector[0, np.random.choice(matrix_midi.shape[0], p=(out/sum).flatten())] = 1
    return vector

def sample(graph):
    '''
    We can generate from a note or from a sequence
    '''
    #from a note
    x = note_to_vector([(45, 80)]) #une note 1,128
    result = graph.generate(identity, x)
    r = np.asarray(result)
    result_tot = np.concatenate((x.T, r.reshape([r.shape[0], 128]).T), axis=1)

    #from a sequence
    '''
    mat = matrix_midi[:,:8000]
    seq = []
    for i in range(mat.shape[1]):
        seq.append(mat[:,i].T.reshape([1,128]))
    result = graph.generateFromSequence(identity, seq)
    r = np.asarray(result)
    seq = np.asarray(seq)
    seq = seq.reshape([seq.shape[0],128]).T
    result_tot = np.concatenate((seq,r.reshape([r.shape[0],128]).T),axis=1)
    '''


    track = matrixToMidi(result_tot)
    return track



if __name__ == '__main__':
    layer = pickle.load(open('models_mz1_midi/30-04-2017 17h15min44s_b-200.pickle', 'rb'))
    graph = RecurrentGraph(layer, len_seq - 1, hidden_shapes)
    track = sample(graph)
    midi_generated = MidiFile()
    track0 = MidiTrack() #We manually add the tempo and the global structure which are not learnt by the network for the moment
    track0.append(MetaMessage('track_name', name='Tempo', time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=100, notated_32nd_notes_per_beat=1,time=0))
    track0.append(MetaMessage('key_signature', key='C', time=0))
    track0.append(MetaMessage('set_tempo', tempo=int(16777215), time=0))
    track0.append(MetaMessage('end_of_track', time=0))
    midi_generated.tracks.append(track0)
    midi_generated.tracks.append(track)
    midi_generated.save('midi_generated19.mid')
    for msg in track:
        print(msg)
