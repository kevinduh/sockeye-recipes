#!/usr/bin/env python

import sys, getopt

class Datapoint:
    """Defines a point in K-dimensional space"""
    def __init__(self,id):
        self.id = id # datapoint id (0,..N-1)
        self.vec = [] # the K-dim vector
        self.paretoStatus = -1 # -1=dont know, 1=pareto, 0=not pareto
        self.dominatedCount = 0 # number of datapoints that dominate this point
        self.dominatingSet = [] # set of vectors this one is dominating

    def addNumber(self,num):
        """Adds a number to one dimension of this datapoint"""
        self.vec.append(num)

    def addToDominatingSet(self,id2):
        """Add id of of dominating point"""
        self.dominatingSet.append(id2)

    def dominates(self,other):
        """Returns true if self[k]>=other[k] for all k and self[k]>other[k] for at least one k"""
        assert isinstance(other,Datapoint)
        gte=0 # count of self[k]>=other[k]
        gt=0 # count of self[k]>other[k]
        for k in range(len(self.vec)):
            if self.vec[k] >= other.vec[k]:
                gte+=1
                if self.vec[k] > other.vec[k]:
                    gt+=1

        return (gte==len(self.vec) and (gt>0))

    def __repr__(self):
        return self.vec.__repr__()+"\t"+str(self.paretoStatus)


def nondominated_sort(dataset):
    """Nondominated Sorting, generates ranking w/ higher number = better pareto front"""
    numPareto = 0

    # pairwise comparisons
    for n in range(len(dataset)):
        for m in range(len(dataset)):
            if dataset[m].dominates(dataset[n]):
                dataset[n].dominatedCount+=1
                dataset[m].addToDominatingSet(n)

    # find first pareto front
    front = []
    front2 = []
    tmpLevel = -10 # temporary value for Pareto level, will re-adjust later
    for n in range(len(dataset)):
        if dataset[n].dominatedCount == 0:
            dataset[n].paretoStatus = tmpLevel
            front.append(n)
            numPareto+=1

    # iteratively peel off pareto fronts
    while len(front) != 0:
        tmpLevel-=1
        for f in front:
            for s in dataset[f].dominatingSet:
                dataset[s].dominatedCount -= 1
                if dataset[s].dominatedCount == 0:
                    front2.append(s)
                    dataset[s].paretoStatus = tmpLevel
        front = front2
        front2 = []

    # re-adjust pareto level
    for n in range(len(dataset)):
        oldLevel = dataset[n].paretoStatus
        if oldLevel != -1:
            dataset[n].paretoStatus = oldLevel-tmpLevel-1

    return numPareto


def create_dataset(raw_vectors):
    """Given a list of vectors, create list of datapoints"""
    dataset = []
    for k in range(len(raw_vectors)):
        for n,v in enumerate(raw_vectors[k]):
            if k == 0:
                dataset.append(Datapoint(n))
            dataset[n].addNumber(v)
    return dataset


def readfile(filename,multiplier=1.0):
    """Reads a vector file (objective values in one dimension)"""
    with open(filename,'r') as f:
        lines = f.readlines()
    vec = [multiplier*float(a.strip().split('\t')[1]) for a in lines]
    return vec


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:],"l:s:")
    except getopt.GetoptError:
        print("pareto.py -l file1 -s file2 -l file3 ...")
        sys.exit(2)

    raw_vectors=[]
    for opt, arg in opts:
        if opt == '-l':
            raw_vectors.append(readfile(arg,1.0))
        elif opt == '-s':
            raw_vectors.append(readfile(arg,-1.0))

    dataset = create_dataset(raw_vectors)
    nondominated_sort(dataset)
    for s in dataset:
        print(s)
