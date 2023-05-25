#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from music21 import converter
from collections import Counter , note , chord
import matplotlib.pyplot as plt


# In[2]:


folder_path = r"C:\Users\Siddharth\Downloads\Spring 2023 U Chicago\MSCA 31009 Machine Learning & Predictive Analytics\Final Project\midi_files"


# In[3]:


# Basic statistics
file_count = len(os.listdir(folder_path))
print("Number of MIDI files:", file_count)


# In[7]:


# Note distribution
note_counts = Counter()
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    midi_stream = converter.parse(file_path)
    for element in midi_stream.flat.notesAndRests:
        if isinstance(element, note.Note):
            note_counts[element.nameWithOctave] += 1
        elif isinstance(element, chord.Chord):
            for note_obj in element.notes:
                note_counts[note_obj.nameWithOctave] += 1


# In[14]:


# Calculate the total overall number of notes
total_notes = sum(note_counts.values())
print("Total number of notes:", total_notes)


# In[15]:


# Count the number of different types of notes
num_different_notes = len(note_counts)
print("Number of different types of notes:", num_different_notes)


# In[10]:


# Time signature exploration
time_signatures = []
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    midi_stream = converter.parse(file_path)
    time_signature = midi_stream.flat.getTimeSignatures()[0]
    time_signatures.append(time_signature.ratioString)

# Print the time signature distribution
print("Time Signature Distribution:")
for signature, count in Counter(time_signatures).most_common():
    print(signature, ":", count)


# In[12]:


# Key signature analysis
key_signatures = []
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    midi_stream = converter.parse(file_path)
    key_signatures_elements = midi_stream.flat.getElementsByClass("KeySignature")
    if key_signatures_elements:
        key_signature = key_signatures_elements[0]
        key_signatures.append(key_signature.sharps)

# Plot the key signature distribution
plt.hist(key_signatures, bins=range(-7, 8))
plt.xlabel("Number of Sharps")
plt.ylabel("Frequency")
plt.title("Key Signature Distribution")
plt.show()

