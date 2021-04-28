
<img src="./data/out_shape.png" alt="shape_processing" width="256" height="412"><img src="./data/out_color.png" alt="shape_processing" width="256" height="412">

# Graph_Signal_Processing
- This project implements *Total Variation* and *Tikhonov* regularization on graphs to process pointclouds.
- It uses `torch_geometric` for fast processing on gpu.
- It is recommended to [render the notebooks here](www.example.com).


# Installation
- One needs `faiss` for the graph construction.
- One needs `torch_geometric` for processing the signal on the graphs.
- Start by installing  [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Activate the conda base environment using `conda activate`.
- Install `pytorch` and `faiss` in the conda environment using 
```
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
$ conda install -c pytorch faiss-gpu cudatoolkit=10.2
```
- You may need to change the cudatookit version. More details on above step are [here](https://pytorch.org/get-started/locally/) and [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).
- The next two steps *are important*.
- Install torch_geometric **inside the conda environment using the pip command, Not the `conda install` command**. Just follow the steps [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
- To render the pointclouds locally, one needs open3d. Again, it is recommended to install open3d using the pip command **Not** the `conda-install` command.
```
(base) $ pip install open3d
```


# Mention all the relevant details
