# Awesome-LoRA
[python-img]: https://img.shields.io/github/languages/top/yuheng2000/Awesome-LoRA?color=lightgrey
[stars-img]: https://img.shields.io/github/stars/yuheng2000/Awesome-LoRA?color=yellow
[stars-url]: https://github.com/yuheng2000/Awesome-LoRA/stargazers
[fork-img]: https://img.shields.io/github/forks/yuheng2000/Awesome-LoRA?color=lightblue&label=fork
[fork-url]: https://github.com/yuheng2000/Awesome-LoRA/network/members
[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=yuheng2000.Awesome-LoRA
[adgc-url]: https://github.com/yuheng2000/Awesome-LoRA



AL is a collection of state-of-the-art (SOTA), novel low-rank adaptation methods (papers, codes and datasets). Any other interesting papers and codes are welcome. Any problems, please contact jiyuheng2023@ia.ac.cn. If you find this repository useful to your research or work, it is really appreciated to star this repository. :sparkles: 

[![Made with Python][python-img]][adgc-url]
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][adgc-url]

--------------

## What's LoRA (Low-Rank Adaptation)?

LoRA is an efficient finetuning technique proposed by Microsoft researchers to adapt large models to specific tasks and datasets.

## Important Survey Papers

| Year | Title                                                        |    Venue    |                            Paper                             | Code |
| ---- | ------------------------------------------------------------ | :---------: | :----------------------------------------------------------: | :--: |
| 2023 | **An Overview of Advanced Deep Graph Node Clustering** |    TCSS   | [Link](https://ieeexplore.ieee.org/abstract/document/10049408) |  - |
| 2022 | **A Survey of Deep Graph Clustering: Taxonomy, Challenge, and Application** |    arXiv    | [Link](https://arxiv.org/abs/2211.12875) |  [Link](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering) |
| 2022 | **A Comprehensive Survey on Community Detection with Deep Learning** |    TNNLS    | [Link](https://arxiv.org/pdf/2105.12584.pdf?ref=https://githubhelp.com) |  -   |
| 2020 | **A Comprehensive Survey on Graph Neural Networks**          |    TNNLS    | [Link](https://ieeexplore.ieee.org/abstract/document/9046288) |  -   |
| 2020 | **Deep Learning for Community Detection: Progress, Challenges and Opportunities** |    IJCAI    |           [Link](https://arxiv.org/pdf/2005.08225)           |  -   |
| 2018 | **A survey of clustering with deep learning: From the perspective of network architecture** | IEEE Access | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8412085) |  -   |



## Papers

### Other Related Methods

| Year | Title                                                        | **Venue** |                            Paper                             |                             Code                             |
| :--: | :----------------------------------------------------------- | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2023 | **GPUSCAN++: Efficient Structural Graph Clustering on GPUs** | arXiv | [Link](https://arxiv.org/pdf/2311.12281.pdf) | - |
| 2022 | **Deep linear graph attention model for attributed graph clustering** | Knowl Based Syst | [Link](https://doi.org/10.1016/j.knosys.2022.108665) | - |
| 2022 | **Scalable Deep Graph Clustering with Random-walk based Self-supervised Learning** | WWW | [Link](https://arxiv.org/pdf/2112.15530) | - |
| 2022 | **X-GOAL: Multiplex Heterogeneous Graph Prototypical Contrastive Learning (X-GOAL)** | arXiv | [Link](https://arxiv.org/pdf/2109.03560) | - |
| 2022 | **Deep Graph Clustering with Multi-Level Subspace Fusion** |   PR    |      [Link](https://doi.org/10.1016/j.patcog.2022.109077)      |-|
| 2022 | **GRACE: A General Graph Convolution Framework for Attributed Graph Clustering** |   TKDD    |      [Link](https://dl.acm.org/doi/pdf/10.1145/3544977)      |                               [Link](https://github.com/BarakeelFanseu/GRACE)                               |                               |
| 2022 | **Fine-grained Attributed Graph Clustering**                 |    SDM    | [Link](https://epubs.siam.org/doi/epdf/10.1137/1.9781611977172.42) |            [Link](https://github.com/sckangz/FGC)            |
| 2022 | **Multi-view graph embedding clustering network: Joint self-supervision and block diagonal representation** |    NN     | [Link](https://www.sciencedirect.com/science/article/pii/S089360802100397X?via%3Dihub) |       [Link](https://github.com/xdweixia/NN-2022-MVGC)       |
| 2022 | **SAGES: Scalable Attributed Graph Embedding with Sampling for Unsupervised Learning** |   TKDE    | [Link](https://ieeexplore.ieee.org/abstract/document/9705119) |                              -                               |
| 2022 | **Automated Self-Supervised Learning For Graphs**            |   ICLR    |     [Link](https://openreview.net/forum?id=rFbR4Fv-D6-)      |       [Link](https://github.com/ChandlerBang/AutoSSL)        |
| 2022 | **Stationary diffusion state neural estimation for multi-view clustering** |   AAAI    |           [Link](https://arxiv.org/abs/2112.01334)           |           [Link](https://github.com/kunzhan/SDSNE)           |
| 2021 | **Simple Spectral Graph Convolution**                        |   ICLR    |      [Link](https://openreview.net/pdf?id=CYO5T-YjWZV)       |         [Link](https://github.com/allenhaozhu/SSGC)          |
| 2021 | **Spectral embedding network for attributed graph clustering (SENet)** |    NN     | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0893608021002227) |                              -                               |
| 2021 | **Smoothness Sensor: Adaptive Smoothness Transition Graph Convolutions for Attributed Graph Clustering** |   TCYB    | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9514513) |           [Link](https://github.com/aI-area/NASGC)           |
| 2021 | **Multi-view Attributed Graph Clustering**                   |   TKDE    | [Link](https://www.researchgate.net/profile/Zhao-Kang-6/publication/353747180_Multi-view_Attributed_Graph_Clustering/links/612059cd0c2bfa282a5cd55e/Multi-view-Attributed-Graph-Clustering.pdf) |           [Link](https://github.com/sckangz/MAGC)            |
| 2021 | **High-order Deep Multiplex Infomax**                        |    WWW    |           [Link](https://arxiv.org/abs/2102.07810)           |          [Link](https://github.com/baoyujing/HDMI)           |
| 2021 | **Graph InfoClust: Maximizing Coarse-Grain Mutual Information in Graphs** |   PAKDD   | [Link](https://link.springer.com/chapter/10.1007%2F978-3-030-75762-5_43) |    [Link](https://github.com/cmavro/Graph-InfoClust-GIC)     |
| 2021 | **Graph Filter-based Multi-view Attributed Graph Clustering** |   IJCAI   |   [Link](https://www.ijcai.org/proceedings/2021/0375.pdf)    |           [Link](https://github.com/sckangz/MvAGC)           |
| 2021 | **Graph-MVP: Multi-View Prototypical Contrastive Learning for Multiplex Graphs** |   arXiv   |           [Link](https://arxiv.org/abs/2109.03560)           |         [Link](https://github.com/chao1224/GraphMVP)         |
| 2021 | **Contrastive Laplacian Eigenmaps**                          |  NeurIPS  | [Link](https://proceedings.neurips.cc/paper/2021/file/2d1b2a5ff364606ff041650887723470-Paper.pdf) |         [Link](https://github.com/allenhaozhu/COLES)         |
| 2020 | **Cluster-Aware Graph Neural Networks for Unsupervised Graph Representation Learning** |   arXiv   |           [Link](https://arxiv.org/abs/2009.01674)           | - |
| 2020 | **Distribution-induced Bidirectional GAN for Graph Representation Learning** |   CVPR    |           [Link](https://arxiv.org/pdf/1912.01899)           |           [Link](https://github.com/SsGood/DBGAN)            |
| 2020 | **Adaptive Graph Converlutional Network with Attention Graph Clustering for Co saliency Detection** |   CVPR    | [Link](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Adaptive_Graph_Convolutional_Network_With_Attention_Graph_Clustering_for_Co-Saliency_CVPR_2020_paper.pdf) |      [Link](https://github.com/ltp1995/GCAGC-CVPR2020)       |
| 2020 | **Spectral Clustering with Graph Neural Networks for Graph Pooling (MinCutPool)** |   ICML    | [Link](http://proceedings.mlr.press/v119/bianchi20a/bianchi20a.pdf) | [Link](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling) |
| 2020 | **MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding** |    WWW    |           [Link](https://arxiv.org/abs/2002.01680)           |          [Link](https://github.com/cynricfu/MAGNN)           |
| 2020 | **Unsupervised Attributed Multiplex Network Embedding**      |   AAAI    |           [Link](https://arxiv.org/abs/1911.06750)           |           [Link](https://github.com/pcy1302/DMGI)            |
| 2020 | **Cross-Graph: Robust and Unsupervised Embedding for Attributed Graphs with Corrupted Structure** |   ICDM    |     [Link](https://ieeexplore.ieee.org/document/9338269)     |      [Link](https://github.com/FakeTibbers/Cross-Graph)      |
| 2020 | **Multi-class imbalanced graph convolutional network learning** | IJCAI | [Link](https://www.ijcai.org/proceedings/2020/0398.pdf) | - |
| 2020 | **CAGNN: Cluster-Aware Graph Neural Networks for Unsupervised Graph Representation Learning** |   arXiv   |   [Link](http://arxiv.org/abs/2009.01674)    |           -            |
| 2020 | **Attributed Graph Clustering via Deep Adaptive Graph Maximization** |   ICCKE   | [Link](https://ieeexplore-ieee-org-s.nudtproxy.yitlink.com/stamp/stamp.jsp?tp=&arnumber=9303694) |                              -                               |
| 2019 | **Heterogeneous Graph Attention Network (HAN)**           |    WWW    |         [Link](https://arxiv.org/pdf/1903.07293.pdf)         |            [Link](https://github.com/Jhy1993/HAN)            |
| 2019 | **Multi-view Consensus Graph Clustering**                    |    TIP    | [Link](https://ieeexplore.ieee.org/abstract/document/8501973) |           [Link](https://github.com/kunzhan/MCGC)            |
| 2019 | **Attributed Graph Clustering via Adaptive Graph Convolution (AGC)** |   IJCAI   |   [Link](https://www.ijcai.org/Proceedings/2019/0601.pdf)    |      [Link](https://github.com/karenlatong/AGC-master)       |
| 2016 | **node2vec: Scalable Feature Learning for Networks (node2vec)** | SIGKDD | [Link](https://dl.acm.org/doi/abs/10.1145/2939672.2939754?casa_token=jt4dhGo-tKEAAAAA:lhscLc-u0XZFYYyi48kXK3_vtYR-PffsbbMRZdtpbaprcB1FGyjWH1RvstHACYALyZ9OtUf2nv_FjQ) | [Link](http://snap.stanford.edu/node2vec/) |
| 2016 | **Variational Graph Auto-Encoders (GAE)** | NeurIPS Workshop | [Link](https://ieeexplore.ieee.org/abstract/document/9046288) | [Link](https://github.com/tkipf/gae) |
| 2015 | **LINE: Large-scale Information Network Embedding (LINE)** | WWW | [Link](https://dl.acm.org/doi/pdf/10.1145/2736277.2741093?casa_token=ahQ9yUhknkAAAAAA:lP6rusbODmZ1ZpGxF-cIiiopMiAA8Q4I02cBBbfE5dc8-NQpiPOdV0cv4-43lA9CkTXU4mPei39UDg) | [Link](https://github.com/tangjianpku/LINE) |
| 2014 | **DeepWalk: Online Learning of Social Representations (DeepWalk)** | SIGKDD | [Link](https://dl.acm.org/doi/pdf/10.1145/2623330.2623732?casa_token=x6Gui_HExYoAAAAA:mzfm0BH0rSX7qcQV2WJ6uTSsg7zjnPalmOQ8sQuoJrwXfh9fcDgVPgXb-APCLGk1qWsPpIkBhI61pw) | [Link](https://github.com/phanein/deepwalk) |




## Benchmark Datasets

We divide the datasets into two categories, i.e. graph datasets and non-graph datasets. Graph datasets are some graphs in real-world, such as citation networks, social networks and so on. Non-graph datasets are NOT graph type. However, if necessary, we could construct "adjacency matrices"  by K-Nearest Neighbors (KNN) algorithm.



#### Quick Start

- Step1: Download all datasets from \[[Google Drive](https://drive.google.com/drive/folders/1thSxtAexbvOyjx-bJre8D4OyFKsBe1bK?usp=sharing) | [Nutstore](https://www.jianguoyun.com/p/DfzK1pwQwdaSChjI2aME)]. Optionally, download some of them from URLs in the tables (Google Drive)
- Step2: Unzip them to **./dataset/**
- Step3: Change the type and the name of the dataset in **main.py**
- Step4: Run the **main.py**



#### Code

- **utils.py**
  1. **load_graph_data**: load graph datasets 
  2. **load_data**: load non-graph datasets
  3. **normalize_adj**: normalize the adjacency matrix
  4. **diffusion_adj**: calculate the graph diffusion
  5. **construct_graph**: construct the knn graph for non-graph datasets
  6. **numpy_to_torch**: convert numpy to torch
  7. **torch_to_numpy**: convert torch to numpy
- **clustering.py**
  1. **setup_seed**:  fix the random seed
  2. **evaluation**: evaluate the performance of clustering
  3. **k_means**: K-means algorithm
- **visualization.py**
  1. **t_sne**: t-SNE algorithm
  2. **similarity_plot**: visualize cosine similarity matrix of the embedding or feature



#### Datasets Details

About the introduction of each dataset, please check [here](./dataset/README.md)

1. Graph Datasets

   | Dataset  | # Samples | # Dimension | # Edges | # Classes |                             URL                              |
   | :------: | :-------: | :---------: | :-----: | :-------: | :----------------------------------------------------------: |
   |   CORA   |   2708    |    1433     |  5278   |     7     | [cora.zip](https://drive.google.com/file/d/1_LesghFTQ02vKOBUfDP8fmDF1JP3MPrJ/view?usp=sharing) |
   | CITESEER |   3327    |    3703     |  4552   |     6     | [citeseer.zip](https://drive.google.com/file/d/1dEsxq5z5dc35tS3E46pg6pc2LUMlF6jF/view?usp=sharing) |
   |   CITE   |   3327    |    3703     |  4552   |     6     | [cite.zip](https://drive.google.com/file/d/1dEsxq5z5dc35tS3E46pg6pc2LUMlF6jF/view?usp=sharing) |
   |  PUBMED  |   19717   |     500     |  44324  |     3     | [pubmed.zip](https://drive.google.com/file/d/1tdr20dvvjZ9tBHXj8xl6wjO9mQzD0rzA/view?usp=sharing) |
   |   DBLP   |   4057    |     334     |  3528   |     4     | [dblp.zip](https://drive.google.com/file/d/1XWWMIDyvCQ4VJFnAmXS848ksN9MFm5ys/view?usp=sharing) |
   |   ACM    |   3025    |    1870     |  13128  |     3     | [acm.zip](https://drive.google.com/file/d/19j7zmQ-AMgzTX7yZoKzUK5wVxQwO5alx/view?usp=sharing) |
   |   AMAP   |   7650    |     745     | 119081  |     8     | [amap.zip](https://drive.google.com/file/d/1qqLWPnBOPkFktHfGMrY9nu8hioyVZV31/view?usp=sharing) |
   |   AMAC   |   13752   |     767     | 245861  |    10     | [amac.zip](https://drive.google.com/file/d/1DJhSOYWXzlRDSTvaC27bSmacTbGq6Ink/view?usp=sharing) |
   | CORAFULL |   19793   |    8710     |  63421  |    70     | [corafull.zip](https://drive.google.com/file/d/1XLqs084J3xgWW9jtbBXJOmmY84goT1CE/view?usp=sharing) |
   |   WIKI   |   2405    |    4973     |  8261   |    17     | [wiki.zip](https://drive.google.com/file/d/1vxupFQaEvw933yUuWzzgQXxIMQ_46dva/view?usp=sharing) |
   |   COCS   |   18333   |    6805     |  81894  |    15     | [cocs.zip](https://drive.google.com/file/d/186twSfkDNmqh9L618iCeWq4DA7Lnpte0/view?usp=sharing) |
   | CORNELL  |    183    |    1703     |   149   |     5     | [cornell.zip](https://drive.google.com/file/d/1EjpHP26Oh0_qHl13vOfEzc4ZyzkGrR-M/view?usp=sharing) |
   |  TEXAS   |    183    |    1703     |   162   |     5     | [texas.zip](https://drive.google.com/file/d/1kpz6b9-OsEU1RsAyxWWeUgzhdd3-koI2/view?usp=sharing) |
   |   WISC   |    251    |    1703     |   257   |     5     | [wisc.zip](https://drive.google.com/file/d/1I8v1H1IthEiWd4IoV-wXNF6g1Wtg_sVC/view?usp=sharing) |
   |   FILM   |   7600    |     932     |  15009  |     5     | [film.zip](https://drive.google.com/file/d/1s5K9Gb235-gO-IwevJLKAts7jExnnmrC/view?usp=sharing) |
   |   BAT    |    131    |     81      |  1038   |     4     | [bat.zip](https://drive.google.com/file/d/1hRPtdFo9CzcxlFb84NWXg-HmViZnqshu/view?usp=sharing) |
   |   EAT    |    399    |     203     |  5994   |     4     | [eat.zip](https://drive.google.com/file/d/1iE0AFKs1V5-nMk2XhV-TnfmPhvh0L9uo/view?usp=sharing) |
   |   UAT    |   1190    |     239     |  13599  |     4     | [uat.zip](https://drive.google.com/file/d/1RUTHp54dVPB-VGPsEk8tV32DsSU0l-n_/view?usp=sharing) |
   

**Edges**: Here, we just count the number of undirected edges.

2. Non-graph Datasets

   | Dataset | Samples | Dimension |  Type  | Classes |                             URL                              |
   | :-----: | :-----: | :-------: | :----: | :-----: | :----------------------------------------------------------: |
   |  USPS   |  9298   |    256    | Image  |   10    | [usps.zip](https://drive.google.com/file/d/19oBkSeIluW3A5kcV7W0UM1Bt6V9Q62e-/view?usp=sharing) |
   |  HHAR   |  10299  |    561    | Record |    6    | [hhar.zip](https://drive.google.com/file/d/126OFuNhf2u-g9Tr0wukk0T8uM1cuPzy2/view?usp=sharing) |
   |  REUT   |  10000  |   2000    |  Text  |    4    | [reut.zip](https://drive.google.com/file/d/12MpPWyN87bu-AQYTyjdEcofy1mgjgzi9/view?usp=sharing) |



## Citation

```
@article{deep_graph_clustering_survey,
  title={A Survey of Deep Graph Clustering: Taxonomy, Challenge, and Application},
  author={Liu, Yue and Xia, Jun and Zhou, Sihang and Wang, Siwei and Guo, Xifeng and Yang, Xihong and Liang, Ke and Tu, Wenxuan and Li, Z. Stan and Liu, Xinwang},
  journal={arXiv preprint arXiv:2211.12875},
  year={2022}
}

@article{SCGC,
  title={Simple contrastive graph clustering},
  author={Liu, Yue and Yang, Xihong and Zhou, Sihang and Liu, Xinwang and Wang, Siwei and Liang, Ke and Tu, Wenxuan and Li, Liang},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}

@inproceedings{Dink_Net,
  title={Dink-net: Neural clustering on large graphs},
  author={Liu, Yue and Liang, Ke and Xia, Jun and Zhou, Sihang and Yang, Xihong and Liu, Xinwang and Li, Stan Z},
  booktitle={Proceedings of International Conference on Machine Learning},
  year={2023}
}

@inproceedings{TGC_ML_ICLR,
  title={Deep Temporal Graph Clustering},
  author={Liu, Meng and Liu, Yue and Liang, Ke and Tu, Wenxuan and Wang, Siwei and Zhou, Sihang and Liu, Xinwang},
  booktitle={The 12th International Conference on Learning Representations},
  year={2024}
}

@inproceedings{HSAN,
  title={Hard sample aware network for contrastive deep graph clustering},
  author={Liu, Yue and Yang, Xihong and Zhou, Sihang and Liu, Xinwang and Wang, Zhen and Liang, Ke and Tu, Wenxuan and Li, Liang and Duan, Jingcan and Chen, Cancan},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={37},
  number={7},
  pages={8914-8922},
  year={2023}
}

@inproceedings{DCRN,
  title={Deep Graph Clustering via Dual Correlation Reduction},
  author={Liu, Yue and Tu, Wenxuan and Zhou, Sihang and Liu, Xinwang and Song, Linxuan and Yang, Xihong and Zhu, En},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={7},
  pages={7603-7611},
  year={2022}
}


@inproceedings{liuyue_RGC,
  title={Reinforcement Graph Clustering with Unknown Cluster Number},
  author={Liu, Yue and Liang, Ke and Xia, Jun and Yang, Xihong and Zhou, Sihang and Liu, Meng and Liu, Xinwang and Li, Stan Z},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={3528--3537},
  year={2023}
}




@article{RGAE,
  title={Rethinking Graph Auto-Encoder Models for Attributed Graph Clustering},
  author={Mrabah, Nairouz and Bouguessa, Mohamed and Touati, Mohamed Fawzi and Ksantini, Riadh},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022}
}
```



## Other Related Awesome Repository

[A Unified Framework for Deep Attribute Graph Clustering](https://github.com/Marigoldwu/A-Unified-Framework-for-Deep-Attribute-Graph-Clustering)

[Awesome Partial Graph Machine Learning](https://github.com/WxTu/Awesome-Partial-Graph-Machine-Learning)

[Awesome Knowledge Graph Reasoning](https://github.com/LIANGKE23/Awesome-Knowledge-Graph-Reasoning)

[Awesome Temporal Graph Learning](https://github.com/MGitHubL/Awesome-Temporal-Graph-Learning)

[Awesome Deep Multiview Clustering](https://github.com/jinjiaqi1998/Awesome-Deep-Multiview-Clustering)



