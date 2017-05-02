from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np
from matplotlib import pyplot as plt


def midiToMatrix(midi_file):
    '''
    Create a matrix from a midi file
    Each column/vector has a length of 128 (128 different notes). It contains the velocity (between 0 and 127) of each note at a tick t.
    The second dimension represents the time in the midi file which is decomposed into ticks.
    '''
    for track in midi_file.tracks:
        total_ticks = 0
        musical_events = []
        for event in track:
            if not event.is_meta:
                total_ticks += event.time
                musical_events.append(event)
        #matrix = np.zeros((128, total_ticks+1))*2/127-1 #Normalization between -1 and 1, use it with a final tanh node
        matrix = np.zeros((128, total_ticks + 1)) / 127  #Normalization between 0 and 1, use it with a final sigmoid node
        #current_vector = np.zeros(128)*2/127-1 #for tanh
        current_vector = np.zeros(128) / 127 #for sigmoid
        position = 0 #position of the next tick
        for event in musical_events:
            if event.time!=0:
                for i in range(event.time):
                    #if current_vector.any():  #in order to consider only tick with at least one note. (no silence considered)
                    matrix[:,position] = current_vector
                    position += 1
            if event.type == 'note_on':
                #current_vector[event.note] = event.velocity*2/127 - 1 #for tanh
                current_vector[event.note] = event.velocity / 127  #for sigmoid
            elif event.type == 'note_off':
                current_vector[event.note] = 0 #0 for sigmoid, -1 for tanh
        j=0
        while position < total_ticks: #complete the matrix until the end of the file
            j+=1
            matrix[:,position] = current_vector
            position += 1
        '''
        #to delete the end of the matrix if there is no note
        i=-1
        print(matrix.shape)
        while -i<matrix.shape[1] and matrix[:,i].any()==False:
            i -= 1
        matrix = matrix[:,:i+1]
        '''
    return matrix


def matrixToMidi(matrix):
    '''
    Create the musical track from a matrix
    We get the velocity of each note at each time from the matrix
    The velocity is chosen with a step of 10 in order to delete the low variations
    '''
    visualize(matrix)  #display the matrix/track
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='Sampler 1', time=0))
    #previous_vector = ((matrix[:,0].reshape([128,1]) + np.ones([128,1]))*127/2)//10*10  %with a final tanh node
    previous_vector = (matrix[:, 0].reshape([128, 1])) * 127 // 10 * 10 #with a final sigmoid node

    for note_index in range(len(previous_vector)):
        if previous_vector[note_index] != 0:
            track.append(Message('note_on', note=note_index, velocity = int(previous_vector[note_index]), time=0))

    delay = 1
    for i in range(1,matrix.shape[1]):
        vector = matrix[:,i]
        if (vector - previous_vector).any()==False:
            delay += 1
        else:
            for i in range(len(vector)):
                #vector[i] = (vector[i] + 1)*127/2 #tanh
                vector[i] = vector[i] * 127  #sigmoid
                vector[i] = vector[i]//10 *10
                if vector[i] != previous_vector[i]:
                    if previous_vector[i]!=0 and vector[i]==0:
                        track.append(Message('note_off', note=i,velocity=0, time=delay))
                    else:
                        track.append(Message('note_on', note=i, velocity=min(int(vector[i]),127),time=delay))
                    delay = 0
            delay += 1
            previous_vector = vector
    track.append(MetaMessage('end_of_track',time=0))
    return track


def visualize(matrix):
    '''
    Display the matrix : the velocity of each note through the time
    '''
    #mat = (matrix + np.ones(matrix.shape))*127/2 //10*10 #with tanh
    mat = matrix * 127 // 10 * 10 #with sigmoid
    print(mat.shape)
    plt.imshow(mat, aspect='auto', vmin=0, vmax=127, cmap='hot')
    plt.colorbar()
    plt.xlabel('Longueur de la séquence')
    plt.ylabel('Notes')
    plt.title('Visualisation du fichier midi')
    plt.show()




if __name__ == '__main__':
    midi_file = MidiFile('music/mz_311_1_format0.mid')
    matrix = midiToMatrix(midi_file)
    outfile = MidiFile()
    track0 = MidiTrack() #We manually add the tempo and the global structure which are not learnt by the network for the moment
    track0.append(MetaMessage('track_name', name='Tempo', time=0))
    track0.append(MetaMessage('time_signature',numerator=4,denominator=4,clocks_per_click=24,notated_32nd_notes_per_beat=8,time=0))
    track0.append(MetaMessage('key_signature',key='C',time=0))
    track0.append(MetaMessage('set_tempo', tempo=int(600000*4.5), time=0))
    track0.append(MetaMessage('end_of_track', time=0))
    track = matrixToMidi(matrix)
    outfile.tracks.append(track0)
    outfile.tracks.append(track)
    outfile.save('test_mz1.mid')
    for msg in track:
        print(msg)