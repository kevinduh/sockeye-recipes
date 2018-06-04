#!/usr/bin/env python
# 
# Given a directory, walk all its subdirectories and delete:
# tensorboard, training_state, mx_optimizer_best.pkl, 
# and all param files (except for param.best)

import os
import sys
import shutil

if (len(sys.argv) != 2):
    print("Usage: python cleanup.py base_directory")
    exit()

basedir=sys.argv[1]
rm_subdir=set(['tensorboard', 'training_state'])
rm_file=set(['mx_optimizer_best.pkl'])

confirm=None
for (dirname, subdirlist, filelist) in os.walk(basedir):

    # check if this is a sockeye-recipes directory
    if 'cmdline.log' in filelist and 'symbol.json' in filelist:

        if confirm != 'a':
            print("Do you want to clean: %s ?" %dirname),
            confirm=raw_input("  [y]es, [n]o, yes to [a]ll ")

        # start removing files if confirmed yes or all
        if confirm == 'y' or confirm == 'a':
            print("Cleaning: %s " %dirname)
            # remove param files
            if 'params.best' in filelist:
                pbest = os.readlink(dirname+'/params.best')
                for f in filelist:
                    if f.startswith('params') and f != pbest and f != 'params.best':
                        os.remove(dirname+"/"+f)
                        print("  rm "+f)

            # remove additional files in rm_file
            for f in filelist: 
                if f in rm_file:
                    os.remove(dirname+"/"+f)
                    print("  rm "+f)

            # remove additional subdirs in rm_subdir
            removed_subdir = []
            for d in subdirlist: 
                if d in rm_subdir:
                    shutil.rmtree(dirname+"/"+d)
                    removed_subdir.append(d)
                    print("  rm "+d)
            for d in removed_subdir:
                subdirlist.remove(d)

        # skip this directory
        else:
            print("Not cleaning %s" %dirname)
                
