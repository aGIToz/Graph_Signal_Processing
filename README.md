# Graph_Processing
Primal Dual algo on graphs 


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
- The next steps are tricky and important.
- Install torch_geometric **inside the conda environment using the pip command Not the `conda install` command**. Just follow the steps here.
- To render the pointcloud one needs open3d. Again, it is recommended to install open3d using the pip command **Not** the `conda-install` command.
```
$ pip install open3d
```


# Mention all the relevant details
