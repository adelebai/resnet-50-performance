# resnet-50-performance
High performance ML project

# Set up 

Needs a kaggle account to run. Generate a kaggle.json file first.

1. Call ```python get_dataset.py``` first to download the dataset


# How to run

To measure training time for data loading or combined results:
1. First modify the config.py for any settings. Recommend running 1 epoch (or even less) if RevNet is enabled.
2. ```python main.py```


To generate tensorboard logs for memory profiling:
1. Edit the profile_memory.py script and select the model you want to profile.
2. ```python profile_memory.py``` to generate a tensorboard log in /logs

# Tensorboard

Store tensorboard charts in /log and visualize them with

```
tensorboard --logdir=./log
```

And go to http://localhost:6006/#pytorch_profiler


# Data Loading Optimization Results

Our results did not indicate any speed improvements from the data loading optimizations. However, this may have been due to the limited scale of our test setup on a single machine, which is not representative of a realistic training setup on petabytes of data. We propose evaluating the described optimizations on a distributed training cluster as a next step.  

 || Data loading time (s) | Forward and backprop time (s) |
 |-----------------------|---------------------------|----|
 |Baseline | 703.08 | 1266.58 |
 |With parallelized DataLoader workers only | 1051.83 | 1139.75 |


# Revnet Optimization Results

Peak Memory Usage with batch size 8:

|Model | Peak Memory Usage (mb) | Time (s) per batch | \# Parameters |
|------------------------|--------------------|---------------|------|
| Adjusted ResNet18 | 743 | 0.3 | 746,436 |
| Adjusted ResNet34 | 1158 | 0.4 | 1,393,476 |
| Adjusted ResNet50 | 3494 | 0.57 | 1,581,700 |
| RevNet 3-3-3 | 466 | 0.4 | 476,020  |
| RevNet 5-5-5 | 468 | 0.84 | 796,468  |
| RevNet 9-9-9 | 470 | 0.98 | 1,437,364  |
| RevNet 18-18-18 | 476 | 1.8 | 2,879,380  |

Note: adjusted means we halved the filter structure of the resnet model, using [32, 32, 64, 112] instead of the regular. This is mostly to reduce parameter count and save time.  

We did observe less memory usage and better memory scaling for deeper RevNet models. Due to time constraints we weren't able to fully measure the convergence performance of the RevNet optimizations and if it truly matches that of a standard ResNet.  

1 epoch convergence:

![alt text](/images/diagram-2.JPG)

# Conclusions

Based on these preliminary results, we wouldn't recommend using RevNet as a memory optimization just due to the sheer additional computational cost it introduces. In fact, with the RevNet optimization the added computation cost outweighed any data-loading speedups introduced by the parallel data workers or memory improvements. 

# References

- The Reversible Residual Network: Backpropagation Without Storing Activations https://arxiv.org/pdf/1707.04585.pdf
- PROFILING AND IMPROVING THE PYTORCH DATALOADER FOR HIGH-LATENCY STORAGE - https://arxiv.org/pdf/2211.04908.pdf 