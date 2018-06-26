#!/usr/bin/env python
# Creates a small file with embeddings for the sockeye-recipes de-en example
# Fasttext embeddings do not contain a meta first line

import sys
import json
import codecs

# TODO: Replace with argparse
vec_filename = sys.argv[1]
# Expected to be JSON
out_vocab_filename = sys.argv[2]
output_filename = sys.argv[3]

out_vocab_fp = codecs.open(out_vocab_filename, encoding="utf8")
# This is a dictionary
out_vocab = json.load(out_vocab_fp)
out_vocab_fp.close()
print("Words in out vocab : " + str(len(out_vocab)))

output_fp = codecs.open(output_filename, "w", encoding="utf8")
words_retained = 0

wr = []

with codecs.open(vec_filename, encoding="utf8") as vec_file:
  for line in vec_file:
    line_comp = line.strip().split()
    word = line_comp[0]
    if word in out_vocab:
      output_fp.write(line)
      wr.append(word)
      words_retained += 1

print("Words retained : " + str(words_retained))
output_fp.close()
