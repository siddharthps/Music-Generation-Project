# Music-Generation-Project

## Introduction
This project is about generating music using LSTM networks. The goal is to train a model on a dataset of midi files and then use the model to generate a music sequence .

## Dataset 

The dataset used for this project is sampled version from the Lakh MIDI dataset V0.1 by Colin Raffel. The full dataset is available at https://colinraffel.com/projects/lmd/.

## Methodology

This project is divided into two parts. In the first part, we train a model on the dataset of midi files. In the second part, we use the trained model to generate a music sequence. 

### Training the model  

For training the model, we use the following steps: 

1. We read the midi files from the dataset and convert them into a list of notes. 
2. We create a sequence of notes from the list of notes and create a vocabulary of unique notes and map them to integers.
3. We create a sequence of input and output pairs from the sequence of notes.

The input to the model is a sequence of notes and the output is the next note in the sequence. We use LSTM networks for training the model.

### The Model 

The model consists of an mutiple LSTM layers with dropouts and Batch Normalization layers for regularization . The model is trained for 150 epochs with a batch size of 128. The model is trained on a GPU and also incorporates early stopping.

### Generating Music

For generating music, we use the following steps: 

1. We randomly select a sequence of notes from the list of notes.
2. We use the trained model to predict the next note in the sequence using the greedy approach.
3. We append the predicted note to the sequence of notes.
4. We use the updated sequence of notes to predict the next note in the sequence and repeat the above steps until we reach the desired length of the sequence.

## Results

The trained LSTM model does generate music but the generated music is not very good. The generated music is not very melodious and the notes are not in sync. The music generated by the model is very repetitive and the model is not able to generate a variety of music.

## Future Work

The model can be improved by using a better dataset. The dataset used for this project is very small and the model can be improved by using a larger dataset. The model can also be improved by using a better architecture. The model can also be improved by using a better approach for generating music. The greedy approach used for generating music is not very good and the model can be improved by using a better approach for generating music.

Transfer Learning can also be used to imporve the model. We can also explore other approaches for generating music such as Reinforcement Learning or GANs. 

## References

1.https://github.com/Skuldur/Classical-Piano-Composer (The basic structure of the code is taken from this repository)
2.https://colinraffel.com/projects/lmd/
