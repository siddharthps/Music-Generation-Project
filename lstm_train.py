#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import pickle
import os
import tensorflow as tf
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation , LSTM, LSTMCell , RNN
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# In[ ]:


# Setting up the path
absolute_directory = 'C://Users//Siddharth//Downloads//Spring 2023 U Chicago//MSCA 31009 Machine Learning & Predictive Analytics//Final Project//'
file_name = 'notes'
absolute_path = os.path.join(absolute_directory, file_name)


# In[ ]:


# Define Function that reads MIDI files and converts them to notes and chords (if applicable) and stores them in a list called notes.
def get_notes():
    notes = []
    # For each file in the midi_songs directory , parse the file and get the notes
    for f in glob.glob('./midi_files/**/*.mid', recursive=True):
        #print('Parsing song: ', f)
        midi = converter.parse(f)
        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts: # if file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # notes are flat stucture
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))               


    os.makedirs(absolute_directory, exist_ok=True)  # Create the 'data' directory if it doesn't exist

    with open(absolute_path, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


# In[ ]:


# Define Function that loads notes from the notes list created above
def load_notes():
    notes = []

    with open(absolute_path, 'rb') as filepath:
        notes = pickle.load(filepath)

    return notes


# In[ ]:


# Define Function that prepares sequences of notes and chords to be fed into the neural network
def prepare_sequences(notes, n_vocab):
    print('Preparing sequences...')

    sequence_length = 20 # Sequence length to be fed into the neural network (i.e. number of notes/chords to be fed into the network at a time)

    # Get pitch names
    pitch_names = sorted(set(n for n in notes))

    # Map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(0, (len(notes) - sequence_length), 1):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]

        seq_in_int = [note_to_int[char] for char in seq_in]
        network_input.append(seq_in_int)

        seq_out_int = note_to_int[seq_out]
        network_output.append(seq_out_int)

    n_patterns = len(network_input)

    # Reshape for LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # Normalize input
    network_input = network_input / float(n_vocab)

    # One-hot encode output
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


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

    return model


# In[ ]:


# Define Function that trains the neural network with early stopping
def train(model, network_input, network_output):
    """Train the neural network with early stopping."""
    print('Training model...')

    # Define filepath for saving the best model weights during training
    filepath = 'weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5'

    # Define model checkpoints and early stopping
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='loss',
        min_delta=0.05,
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True
    )

    # Create a list of callbacks
    callbacks_list = [checkpoint, early_stopping]

    # Train the model with early stopping and model checkpoints
    model.fit(
        network_input,
        network_output,
        epochs=150,
        batch_size=128,
        callbacks=callbacks_list
    )


# In[ ]:


# Define Function that trains the neural network with early stopping
def train_network():
    
    notes = load_notes()

    # Number of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


# In[ ]:


if __name__ == '__main__':
    train_network()

