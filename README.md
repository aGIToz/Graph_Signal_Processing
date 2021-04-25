# Graph_Processing
Primal Dual algo on graphs 


# Installation
- One needs `faiss` for graph construction.
- One needs `torch_geometric` for processing the signal on the graphs.
- Start by installing the Miniconda.
- Activate the conda base environment using `conda activate`
- Install torch and faiss in the conda environment using 
```
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
$ conda install -c pytorch faiss-gpu cudatoolkit=10.2
```
- You may need to change the cudatookit version. More details on above step are here and here.
- The next steps are tricky and important.
- Install torch_geometric **inside the conda environment using the pip command Not the `conda install` command**. Just follow the steps here.
- To render the pointcloud one need open3d. Again, it is very recommended to install the open3d using the pip command **Not** the `conda-install` command.
```
$ pip install open3d
```


# Mention all the relevant details
