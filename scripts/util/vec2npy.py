#!/usr/bin/env python
# Converts fasttext embeddings to the npy format
# Writes out modified embedding file and a vocab in JSON format
# Fasttext embeddings do not contain a meta first line

import sys
import json
import codecs
import numpy as np

# TODO: Replace with argparse
vec_filename = sys.argv[1]
output_prefix = sys.argv[2]

emb_matrix = []
vocab = []

with codecs.open(vec_filename, encoding="utf8") as vec_file:
  for line in vec_file:
    line = line.strip().split()
    vocab.append(line[0])
    emb_matrix.append(np.asarray(line[1:], dtype=np.float32).reshape(-1, 1))

emb_matrix = np.concatenate(emb_matrix, axis=1)
emb_matrix = np.transpose(emb_matrix)
# Set mean=0 per feature
old_mean = np.mean(emb_matrix)
emb_matrix -= np.mean(emb_matrix, axis=0)
print("Rescaled mean of emb matrix (per feature) from " + str(old_mean) + " to " + str(np.mean(emb_matrix)))
print("Shape of emb matrix = " + str(emb_matrix.shape))

filtered_vocab = {vocab[idx] : idx for idx in range(len(vocab))}
json.dump(filtered_vocab, codecs.open(output_prefix + ".vocab", "w", encoding="utf8"))

np.save(output_prefix + ".npy", emb_matrix)
