#!/bin/sh

wget https://cs.jhu.edu/~kevinduh/j/sample-de-en.tgz
tar -xzvf sample-de-en.tgz
rm sample-de-en.tgz

cd sample-de-en
mkdir emb
cd emb
wget https://cs.jhu.edu/~gkumar/data/small.cln.deen.vec.tar.gz
tar -xvzf small.cln.deen.vec.tar.gz
rm small.cln.deen.vec.tar.gz
cd ../..
