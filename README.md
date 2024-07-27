## SCUBa-Net: Spatially-constrained and -unconstrained bi-graph interaction network for multi-organ pathology image classification

Implementation of SCUBa-Net (IEEE TMI)

**Abstract:** Multi-class cancer classification is a crucial task in computational pathology, contributing significantly to healthcare services. Most computational pathology tools have been developed based on convolutional neural networks, Transformers, and graph neural networks. Combinations of these methods have often been shown to improve the classification performance. However, the combination of graph neural networks and Transformers and their relationships have not been well studied and explored.
In this study, we propose a parallel, bi-graph neural network, designated as SCUBa-Net, equipped with both graph convolutional networks and Transformers, that processes a pathology image as two distinct graphs, including a spatially-constrained graph and a spatially-unconstrained graph. For efficient and effective graph learning, we introduce two inter-graph interaction blocks and an intra-graph interaction block. The inter-graph interaction blocks learn the node-to-node interactions within each graph. The intra-graph interaction block learns the graph-to-graph interactions at both global- and local-levels with the help of the virtual nodes that collect and summarize the information from the entire graphs. SCUBa-Net is systematically evaluated on four multi-organ datasets, including colorectal, prostate, gastric, and bladder cancers. The experimental results demonstrate the effectiveness of SCUBa-Net in comparison to the state-of-the-art convolutional neural networks, Transformer, and graph neural networks.

## To-do list

- [x] Update the implementation of SCUBa-Net.
- [x] Update datasets used in the study: KBSMC colon datasets, UHU, UBC prostate datasets, gastric and bladder datasets.
- [ ] Document for constructing spatially-contrained graph.
- [ ] Document for training SCUBa-Net.
- [ ] Document for inference with SCUBa-Net.

## Datasets

![image](https://github.com/user-attachments/assets/551c315d-7aa7-4491-b4d8-374a49487caa)

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
