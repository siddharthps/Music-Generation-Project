#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import pickle
import os
import tensorflow as tf
import numpy as np
from music21 import converter, instrument, note, chord , stream
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation , LSTM, LSTMCell , RNN
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint , EarlyStopping


# In[ ]:


absolute_directory = 'C://Users//Siddharth//Downloads//Spring 2023 U Chicago//MSCA 31009 Machine Learning & Predictive Analytics//Final Project//data'
file_name = 'notes'
absolute_path = os.path.join(absolute_directory, file_name)


# In[ ]:


def generate():
    with open(absolute_path, 'rb') as f:
        notes = pickle.load(f)

    # Get pitch names
    pitchnames = sorted(set(notes))
    n_vocab = len(set(notes))    
     
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)


# In[ ]:


# Define Function that prepares sequences of notes and chords to be fed into the neural network
def prepare_sequences(notes, pitchnames, n_vocab):
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        output.append(note_to_int[seq_out])

    n_patterns = len(network_input)

    # Reshape
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)


# In[ ]:


# Define Function that creates the neural network
def create_network(network_input, n_vocab):
    print('Creating network...')

    model = Sequential()

    model.add(RNN(LSTMCell(1024), input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(RNN(LSTMCell(512), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())    
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Load trained weights
    model.load_weights('weights-improvement-multi-bigger.hdf5')

    return model


# In[ ]:


def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from neural net based on input sequence of notes. """
    print('Generating notes...')

    # Pick random sequence from input as starting point
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # Generate 500 notes
    n = 500
    for note_index in range(n):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        #prediction_input = prediction_input / float(n_vocab) # There is no need to normalize here, as it has already been done in the training set

        prediction = model.predict(prediction_input, verbose=0)

        # Take most probable prediction, convert to note, append to output
        # This is the "greedy" approach to composing music - just take the most likely next note
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        # Convert result note back to scaled representation
        scaled_result = index / float(n_vocab)

        # Scoot input over by 1 note
        pattern.append(scaled_result)
        pattern = pattern[1:len(pattern)]

    return prediction_output


# In[ ]:


def create_midi(prediction_output):
    print('Creating midi...')
    """ Convert prediction output to notes. Create midi file!!!! """
    offset = 0
    output_notes = []
    # Possible extension: multiple/different instruments!
    stored_instrument = instrument.Piano()

    # Create Note and Chord objects
    for pattern in prediction_output:
        # Pattern is a Chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = stored_instrument
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else: # Pattern is a note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = stored_instrument
            output_notes.append(new_note)

        # Increase offset for note
        # Possible extension: ~ RHYTHM ~
        offset += 0.25

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output_song.mid')


# In[ ]:


if __name__ == '__main__':
    generate()

