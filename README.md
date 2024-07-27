## SCUBa-Net: Spatially-constrained and -unconstrained bi-graph interaction network for multi-organ pathology image classification

Implementation of SCUBa-Net (IEEE TMI)

**Abstract:** Multi-class cancer classification is a crucial task in computational pathology, contributing significantly to healthcare services. Most computational pathology tools have been developed based on convolutional neural networks, Transformers, and graph neural networks. Combinations of these methods have often been shown to improve the classification performance. However, the combination of graph neural networks and Transformers and their relationships have not been well studied and explored.
In this study, we propose a parallel, bi-graph neural network, designated as SCUBa-Net, equipped with both graph convolutional networks and Transformers, that processes a pathology image as two distinct graphs, including a spatially-constrained graph and a spatially-unconstrained graph. For efficient and effective graph learning, we introduce two inter-graph interaction blocks and an intra-graph interaction block. The inter-graph interaction blocks learn the node-to-node interactions within each graph. The intra-graph interaction block learns the graph-to-graph interactions at both global- and local-levels with the help of the virtual nodes that collect and summarize the information from the entire graphs. SCUBa-Net is systematically evaluated on four multi-organ datasets, including colorectal, prostate, gastric, and bladder cancers. The experimental results demonstrate the effectiveness of SCUBa-Net in comparison to the state-of-the-art convolutional neural networks, Transformer, and graph neural networks.

## To-do list

- [x] Update the implementation of SCUBa-Net.
- [ ] Update datasets used in the study: KBSMC colon datasets, UHU, UBC prostate datasets, gastric and bladder datasets.
- [ ] Document for constructing spatially-contrained graph.
- [ ] Document for training SCUBa-Net.
- [ ] Document for inference with SCUBa-Net.
