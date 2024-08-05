# Experiments with Looped Transformer from Yang et al.
The code of this repo is based on https://github.com/Leiay/looped_transformer. You can find the paper in [arxiv](https://arxiv.org/abs/2311.12424).


## Overview

Describe the results and work: 

  1) Optimization of training
  2) Probing
  3) Dilation - does it allow to boost performance ?


<p align="center" width="100%">
    <img width="100%" src="figure.png">
</p>


## Setup
Please install and activate the environment through
```shell
conda env create -f environment.yml
conda activate loop_tf
```

## Running Experiments
- For standard transformer training, refer to and execute  `bash exec/script_baseline.sh`.
- For looped transformer training, refer to and execute `bash exec/script_loop.sh`.
  - The parameter `b` determines the maximum loop iteration during training.
  - The parameter `T` sets the loop window size.
- To probe a trained model, refer to and execute `bash exec/script_probe.sh`.
- To work with the OpenML dataset for both standard and looped transformers, refer to and execute `bash exec/script_openml.sh`.
- To plot and compare with baseline methods, refer to notebooks in the `jupyter_notebooks` folder.
