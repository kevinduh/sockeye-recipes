(1) Ensure that ``tensorboard`` is installed in your conda environment.

```bash
source activate sockeye_cpu  
which tensorboard
```
should return a location within your conda environment (It is okay to use a version installed outside but best to use the one installed with sockeye-recipes for build compatibility). E.g.,

```bash
/home/gkumar/miniconda3/envs/sockeye_cpu/bin/tensorboard
```

(2) For the CLSP machines, it is likely that the tensorboard build will not work on the login node. 

```bash
$ tensorboard
2019-02-04 15:57:39.789127: F tensorflow/core/platform/cpu_feature_guard.cc:37] The TensorFlow library was compiled to use SSE4.1 instructions, but these aren't available on your machine.
Aborted
```

ssh to machine where this does work (any a,b,c node) and start the tensorboard process on the rootdir/logdir of the experiment directory you wish to view.

```bash
tensorboard --logdir $LOGDIR
```

You may change the default port (6006) this process is attached to by using the ``--port`` switch. See ``tensorbord --help`` for more details. 

(3) Setup an SSH tunnel from your local machine to the host which is running tensorboard via the login node.
For example, if a (free) local port is 9999 and the host running tensorboard is b13 on port 6006, then run:

```bash
ssh -L 9999:b13:6006 -N <username>@login.clsp.jhu.edu
```

There are many caveats to using this form of tunneling but it should work assuming all ports are free. See <https://superuser.com/questions/96489/an-ssh-tunnel-via-multiple-hops> for more information.

(4) On your local browser navigate to <http://localhost:9999> to view the tensorboard console (assuming the local port you chose was 9999). 