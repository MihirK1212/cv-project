# Paca-ViT
This project is a replication of the [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Grainger_PaCa-ViT_Learning_Patch-to-Cluster_Attention_in_Vision_Transformers_CVPR_2023_paper.pdf) PaCa-ViT: Learning Patch-to-Cluster Attention in Vision Transformer with [code](https://github.com/iVMCL/PaCaViT).

In this, clusters are learned end-to-end, leading to better tokenizers and inducing joint clustering-for-attention and attention-for-clustering for better and interpretable models. This reduces complexity, facilitating a better visual tokenizer and enabling simple forward explainability.

## Novelty
In the above paper, clusters are made using MLP clustering. We introduced many different clustering techniques:-
* K-means clustering
* Hierarchical clustering
* DBSCAN
* Gaussian Mixture Model

To understand how different clustering techniques are used for different tasks, we tested the accuracy of image classification on the CIFAR-10 dataset using the above clustering techniques integrated with PaCa-ViT.

## Results
![image](https://github.com/MihirK1212/cv-project/assets/79632719/0cac7316-d174-40ea-b5fa-2fe4efbe8e5d)

## Installation and Training

## Contributors
* Amit Kumar Makkad
* Mihir Karandikar
* Mukul Jain
* Nilay Ganvit

This project is part of the course CS419 Computer Vision at IIT Indore under the guidance of Dr. Puneet Gupta.
