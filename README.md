## SCUBa-Net: Spatially-constrained and -unconstrained bi-graph interaction network for multi-organ pathology image classification

Implementation of SCUBa-Net (IEEE TMI)

**Abstract:** Multi-class cancer classification is a crucial task in computational pathology, contributing significantly to healthcare services. Most computational pathology tools have been developed based on convolutional neural networks, Transformers, and graph neural networks. Combinations of these methods have often been shown to improve the classification performance. However, the combination of graph neural networks and Transformers and their relationships have not been well studied and explored.
In this study, we propose a parallel, bi-graph neural network, designated as SCUBa-Net, equipped with both graph convolutional networks and Transformers, that processes a pathology image as two distinct graphs, including a spatially-constrained graph and a spatially-unconstrained graph. For efficient and effective graph learning, we introduce two inter-graph interaction blocks and an intra-graph interaction block. The inter-graph interaction blocks learn the node-to-node interactions within each graph. The intra-graph interaction block learns the graph-to-graph interactions at both global- and local-levels with the help of the virtual nodes that collect and summarize the information from the entire graphs. SCUBa-Net is systematically evaluated on four multi-organ datasets, including colorectal, prostate, gastric, and bladder cancers. The experimental results demonstrate the effectiveness of SCUBa-Net in comparison to the state-of-the-art convolutional neural networks, Transformer, and graph neural networks.

![image](https://github.com/user-attachments/assets/11e8cf93-6b2f-46cb-a084-a251567ef600)

## Datasets

1. Colon dataset:
    - KBSMC Colon dataset: [CTrain, CValid & CTest-I](https://drive.google.com/file/d/1KsLvqNdwAnw_WunVyOqi-TIF77BTsn8K/view?usp=sharing)
    - KBSMC independent test set: [CTest-II](https://drive.google.com/file/d/1taYhjlHydhe6TMn4f5J5Lz9SJ-b0IQeS/view).

2. Prostate dataset:
    - UHU Prostate dataset: [PTrain, PValid & PTest-I](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP)
    - UBC Prostate dataset (Independent test set): [PTest-II](https://gleason2019.grand-challenge.org/)

3. Gastric dataset:
    - KBSMC Gastric dataset: [GTrain, GValid, GTest](https://github.com/QuIIL/KBSMC_gastric_cancer_grading_dataset)

4. Bladder dataset:
    - NMI Bladder dataset: [BTrain, BValid, BTest](https://github.com/zizhaozhang/nmi-wsi-diagnosis)

## Obtain node embeddings for spatially-constrained graph

We provide the process to obtain the node embeddings spatially-constrained graph, as described in Section III. Methodology (B.2) in `spatially_constrained_graph` folder:

1. First, an patch image sould be tiled into sub-patches:

```
python 1.crop_sub_patch_images.py --data_path /path/to/dataset --data_output_path /path/to/output
```

2. Then, modify the paths in `config_clr.yaml` to train the SimCLR model for EfficientNetB0 to obtain the embedding network **H^c** (only trained on training set):

```
python 2.train_clr.py
```

3. Obtain node embeddings for the spatially-constrained graph:

```
python 3.build_graphs.py --data_path /path/to/dataset --data_graph_path /path/to/output/graph
```

## Training SCUBa-Net

Use the script `train.py` and specify the directory path of the dataset images (to build the spatially-unconstrained graph), and the path of the node embeddings mentioned above to build the spatially-constrained graph.

```
CUDA_VISIBLE_DEVICES=[gpu id] python train.py --image_path /path/to/dataset --spatially_constrained_graph_path /path/to/output/graph
```

## Evaluating SCUBa-Net

Specify the directory path of the test dataset images as well as the corresponding node embeddings:

```
CUDA_VISIBLE_DEVICES=[gpu id] python test.py --image_path /path/to/dataset --spatially_constrained_graph_path /path/to/output/graph
```

## Citation

If any part of this code is used, please give appropriate citation to our paper. <br />

BibTex entry: <br />
```
@article{Bui2024scubanet,
title = {Spatially-constrained and -unconstrained bi-graph interaction network for multi-organ pathology image classification},
journal = {},
volume = {},
pages = {},
year = {2024},
issn = {},
author = {Doanh C. Bui and Boram Song and Kyungeun Kim and Jin Tae Kwak},
}
```
