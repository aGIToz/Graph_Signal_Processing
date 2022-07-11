
<img src="./data/out_shape.png" alt="shape_processing" width="256" height="412"><img src="./data/out_color.png" alt="shape_processing" width="256" height="412">

# Real world examples:
The following are the examples of denoising on real-world scans of Monument a Napoleon (Cherbourg) and Niki's serpent. First, non-squared data fidelity is used to remove the strong spatial outliers, and then isotropic TV filtering is used on the color channels. The time taken to process Napoleon pointcloud (500k points) is 37 seconds on NVIDIA 1080Ti.
![image](https://user-images.githubusercontent.com/38216671/178206254-101257c8-350f-4c9a-8c51-33ae21da8475.png)
![image](https://user-images.githubusercontent.com/38216671/178206907-d0b7cfd4-b7e9-4b65-bdff-cc2b5670b7cf.png)
![image](https://user-images.githubusercontent.com/38216671/178206752-62d66fdc-f737-4213-b85e-2c808205cd0b.png)

# Graph_Signal_Processing
- This project implements *Total Variation* and *Tikhonov* regularization on graphs to process pointclouds.
- <img src="https://latex.codecogs.com/gif.latex?f^{*}&space;=&space;\underset{f}{\arg\min}&space;\&space;\tfrac{1}{2}\|f-f_0\|^2&space;&plus;&space;\lambda&space;\|\nabla_\omega&space;f\|_{1,p}^{q}" title="f^{*} = \underset{f}{\arg\min} \ \tfrac{1}{2}\|f-f_0\|^2 + \lambda \|\nabla_\omega f\|_{1,p}^{q}" />
- It uses `torch_geometric` for fast processing on gpu.
- It is recommended to [render the notebooks here](https://nbviewer.jupyter.org/github/agitoz/Graph_Signal_Processing/tree/main/?flushed_cache=true).


# Installation
- One needs `faiss` for the graph construction.
- One needs `torch_geometric` for processing the signal on the graphs.
- Start by installing  [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Activate the conda base environment using `conda activate`.
- Install `pytorch` and `faiss` in the conda environment using 
```
(base) $ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
(base) $ conda install -c pytorch faiss-gpu cudatoolkit=10.2
```
- You may need to change the cudatookit version. More details on above step are [here](https://pytorch.org/get-started/locally/) and [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).
- The next two steps *are important*.
- Install torch_geometric **inside the conda environment using the pip command, Not the `conda install` command**. Just follow the steps [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
- To render the pointclouds locally, one needs open3d. Again, it is recommended to install open3d using the pip command **Not** the `conda-install` command.
```
(base) $ pip install open3d
```
# Cite
- The *Primal-Dual* algo implemented in this repo is the same as the one used to produce the results in the Section2 of [this paper](https://arxiv.org/abs/2103.16337).
- The paper argues that some variational-methods can be formulated as a Message Passing Networks which allows the use of graph machine learning libraries (like *torch_geo*, *dgl*) to solve inverse problems on graphs.
```latex
@misc{azad2021variational,
      title={Variational models for signal processing with Graph Neural Networks}, 
      author={Amitoz Azad and Julien Rabin and Abderrahim Elmoataz},
      year={2021},
      eprint={2103.16337},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

