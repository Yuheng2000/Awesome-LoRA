[python-img]: https://img.shields.io/github/languages/top/yuheng2000/Awesome-LoRA?color=lightgrey
[stars-img]: https://img.shields.io/github/stars/yuheng2000/Awesome-LoRA?color=yellow
[stars-url]: https://github.com/yuheng2000/Awesome-LoRA/stargazers
[fork-img]: https://img.shields.io/github/forks/yuheng2000/Awesome-LoRA?color=lightblue&label=fork
[fork-url]: https://github.com/yuheng2000/Awesome-LoRA/network/members
[visitors-img]: https://badges.pufler.dev/visits/yuheng2000/Awesome-LoRA
[adgc-url]: https://github.com/yuheng2000/Awesome-LoRA

## Awesome-LoRA

Awesome-LoRA is a collection of state-of-the-art (SOTA), novel low-rank adaptation methods (papers, codes and datasets). Any other interesting papers and codes are welcome. Any problems, please contact jiyuheng2023@ia.ac.cn. If you find this repository useful to your research or work, it is really appreciated to star this repository. :sparkles: 

[![Made with Python][python-img]][adgc-url]
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][adgc-url]

--------------

## What's LoRA (Low-Rank Adaptation)?

LoRA is an efficient finetuning technique proposed by Microsoft researchers to adapt large models to specific tasks and datasets.

## The pioneering paper

| Year | Title                                                        |    Venue    |                            Paper                             | Code |
| ---- | ------------------------------------------------------------ | :---------: | :----------------------------------------------------------: | :--: |
| 2022 | **LoRA: Low-Rank Adaptation of Large Language Models** |    ICLR   | [Link](https://arxiv.org/abs/2106.09685) |  [Link](https://github.com/microsoft/LoRA) |

## Important Survey Papers

| Year | Title                                                        |    Venue    |                            Paper                             | Code |
| ---- | ------------------------------------------------------------ | :---------: | :----------------------------------------------------------: | :--: |
| 2024 | **A Survey on LoRA of Large Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2407.11046) |  - |



## Papers

| Year | Title                                                        | **Venue** |                            Paper                             |                             Code                             |     **Keywords**     |
| :--: | :----------------------------------------------------------- | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-------: |
| 2024 | **PC-LoRA: Low-Rank Adaptation for Progressive Model Compression with Knowledge Distillation** |    arXiv   | [Link](https://arxiv.org/pdf/2406.09117) | - | Model Compression; Knowledge Distillation; |
| 2024 | **Vera: Vector-based random matrix adaptation** |    ICLR   | [Link](https://arxiv.org/abs/2310.11454) | - | Shared-LoRA; Trainable Vectors; |
| 2024 | **LaMDA: Large Model Fine-Tuning via Spectrally Decomposed Low-Dimensional Adaptation** |    arXiv   | [Link](https://arxiv.org/pdf/2406.12832) | [Link](https://github.com/ArminAzizi98/LaMDA) | Multi-Step Training; Trainable Vectors;|
| 2024 | **Prompt Tuning Strikes Back: Customizing Foundation Models with Low-Rank Prompt Adaptation** |    arXiv   | [Link](https://arxiv.org/pdf/2405.15282) | - | Prompt-Tuning-based |
| 2024 | **ROSA: Random Subspace Adaptation for Efficient Fine-Tuning** |    arXiv   | [Link](https://arxiv.org/pdf/2407.07802) | [Link](https://github.com/rosa-paper/rosa) | Random Subspace Adaptation; Robust Fine-Tuning |
| 2024 | **LoRA-GA: Low-Rank Adaptation with Gradient Approximation** |    arXiv   | [Link](https://arxiv.org/pdf/2407.05000) | [Link](https://github.com/Outsider565/LoRA-GA) | Gradient Approximation; Convergence;|
| 2024 | **Efficient Pareto Manifold Learning with Low-Rank Structure** |    ICML   | [Link](https://openreview.net/pdf?id=a2uFstsHPb) | - | Multi-task learning; Pareto front;|
| 2024 | **AutoLoRa: An Automated Robust Fine-Tuning Framework** |    ICLR   | [Link](https://openreview.net/pdf?id=09xFexjhqE) |  [Link](https://github.com/GodXuxilie/RobustSSL_Benchmark) | Robust Fine-Tuning; Adversarial Robustness;|
| 2024 | **LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters** |    arXiv   | [Link](https://arxiv.org/pdf/2405.17604) |  [Link](https://github.com/MohammadrezaBanaei/LoRA-XS) | scaling language models; SVD； |
| 2024 | **Matrix-Transformation Based Low-Rank Adaptation (MTLoRA): A Brain-Inspired Method for Parameter-Efficient Fine-Tuning** |    arXiv   | [Link](https://arxiv.org/pdf/2403.07440) |  [Link](https://github.com/YaoLiang-Code/MTLoRA-main) | LPLMs; Geometric Structure； |
| 2024 | **AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning** |    arXiv   | [Link](https://arxiv.org/pdf/2403.09113) |  [Link](https://anonymous.4open.science/r/AutoLoRA) | Meta Learning; Rank-1 Matrix|
| 2024 | **RankAdaptor: Hierarchical Dynamic Low-Rank Adaptation for Structural Pruned LLMs** |    arXiv   | [Link](https://arxiv.org/pdf/2406.15734) |  - | Structural Pruning; Hierarchical Dynamic Rank Scheduling |
| 2024 | **LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models** |    arXiv   | [Link](https://arxiv.org/pdf/2403.11627) |  [Link](https://github.com/Young98CN/LoRA_Composer) | Multiconcept Customization; Concept Injection Constraints |
| 2024 | **Investigating Training Strategies and Model Robustness of Low-Rank Adaptation for Language Modeling in Speech Recognition** |    arXiv   | [Link](https://arxiv.org/pdf/2401.10447) |  - | Memory-Efficient Learning; Robust Speech Recognition |
| 2024 | **PRILoRA: Pruned and Rank-Increasing Low-Rank Adaptation** |    arXiv   | [Link](https://arxiv.org/pdf/2401.11316) |  - | Pruned and Rank-Increasing |
| 2024 | **LAMPAT: Low-Rank Adaption for Multilingual Paraphrasing Using Adversarial Training** |    AAAI   | [Link](https://arxiv.org/pdf/2401.04348) |  [Link](https://github.com/VinAIResearch/LAMPAT) | Unsupervised Multilingual Paraphrasing |
| 2024 | **LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2402.11417) |  [Link](https://github.com/yifanycc/loretta) | Tensor-Train Decomposition; Robust Fine-Tuning|
| 2024 | **Derivative-Free Optimization for Low-Rank Adaptation in Large Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2403.01754) |  [Link](https://github.com/stan-anony/derivative_free_lora_rank) | Enhance Robustness; Derivative-Free Optimization |
| 2024 | **LORS: Low-rank Residual Structure for Parameter-Efficient Network Stackingg** |    CVPR   | [Link](https://arxiv.org/pdf/2403.04303) |  - | Reduce Stacking Depth |
| 2024 | **FedLoRA: When Personalized Federated Learning Meets Low-Rank Adaptation** |    ICLR   | [Link](https://openreview.net/pdf?id=bZh06ptG9r) |  [Link](https://github.com/yyn21/SA-FedLora) | Personalized Federated Learning; Data Heterogeneity |
| 2024 | **InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning** |    CVPR   | [Link](https://arxiv.org/pdf/2404.00228) |  [Link](https://github.com/liangyanshuo/InfLoRA) | Continual Learning; Interference-Free |
| 2024 | **AdvLoRA: Adversarial Low-Rank Adaptation of Vision-Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2404.13425) |  - | Adversarial Adaptation; Improve the Robustness|
| 2024 | **Compressible Dynamics in Deep Overparameterized Low-Rank Learning & Adaptation** |    ICML   | [Link](https://arxiv.org/pdf/2406.04112) |  [Link](https://github.com/cjyaras/deep-lora-transformers) | Inherent Low-dimensional Structures of Data; Compressible Dynamics within The Model Parameters; Overparameterization |
| 2024 | **FLORA: Low-Rank Adapters Are Secretly Gradient Compressors** |    ICML   | [Link](https://arxiv.org/pdf/2402.03293) |  [Link](https://github.com/BorealisAI/flora-opt) | High-Rank Updates; Sublinear Space Complexity of Optimization States |
| 2024 | **MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning** |    arXiv   | [Link](https://arxiv.org/pdf/2406.09044) |  - | Robust Fine-Tuning; Adversarial Robustness |
| 2024 | **Expressive and Generalizable Low-rank Adaptation for Large Models via Slow Cascaded Learning** |    arXiv   | [Link](https://arxiv.org/pdf/2407.01491) |  [Link](https://github.com/microsoft/LoRASC) | Cascaded Learning Strategy; Robust Fine-Tuning |
| 2024 | **LoraRetriever: Input-Aware LoRA Retrieval and Composition for Mixed Tasks in the Wild** |    arXiv   | [Link](https://arxiv.org/pdf/2402.09997) |  - | Retrieval and Composition; Mixed Tasks | 
| 2024 | **Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models** |    arXiv   | [Link](https://arxiv.org/pdf/2402.02347) |  [Link](https://github.com/pilancilab/Riemannian_Preconditioned_LoRA) | R×R Preconditioner; Robust Fine-Tuning |
| 2024 | **CoLoRA: Continuous low-rank adaptation for reduced implicit neural modeling of parameterized partial differential equations** |    arXiv   | [Link](https://arxiv.org/pdf/2402.14646) | [Link](https://github.com/julesberman/CoLoRA) | Predicting Speed; Robust Fine-Tuning |
| 2024 | **CorDA: Context-Oriented Decomposition Adaptation of Large Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2406.05223) | [Link](https://github.com/iboing/CorDA) | Context-Oriented Decomposition; Robust Fine-Tuning |
| 2024 | **LoRAP: Transformer Sub-Layers Deserve Differentiated Structured Compression for Large Language Models** |    ICML   | [Link](https://arxiv.org/pdf/2404.09695) |  - | Low-Rank Matrix Approximation; Structured Pruning |
| 2024 | **Asymmetry in Low-Rank Adapters of Foundation Models** |    arXiv   | [Link](https://arxiv.org/pdf/2402.16842) | [Link](https://github.com/Jiacheng-Zhu-AIML/AsymmetryLoRA) | Unexpected Asymmetry In the Importance of Low-Rank Adapter Matrices |
| 2024 | **SAML: Speaker Adaptive Mixture of LoRA Experts for End-to-End ASR** |    arXiv   | [Link](https://arxiv.org/pdf/2406.19706) | [Link](https://github.com/qmgzhao/SAML) | Mixture-Of-Experts(MoE); Speaker Adaptation |
| 2024 | **Dataset Size Recovery from LoRA Weights** |    arXiv   | [Link](https://arxiv.org/pdf/2406.19395) |  [Link](https://github.com/MoSalama98/DSiRe) | Dataset Size Recovery |
| 2024 | **Towards Federated Low-Rank Adaptation with Rank-Heterogeneous Communication** |    arXiv   | [Link](https://arxiv.org/pdf/2406.17477) |  - | Replication-Based Padding Strategy; Federated Learning |
| 2024 | **Retrieval-Augmented Mixture of LoRA Experts for Uploadable Machine Learning** |    arXiv   | [Link](https://arxiv.org/pdf/2406.16989) |  - | Heterogeneous Requests; Uploadable Machine Learning (UML) |
| 2024 | **Bayesian-LoRA: LoRA based Parameter Efficient Fine-Tuning using Optimal Quantization levels and Rank Values trough Differentiable Bayesian Gates** |    arXiv   | [Link](https://arxiv.org/pdf/2406.13046) |  - | Bayesian; Robust Fine-Tuning |
| 2024 | **Mixture-of-Subspaces in Low-Rank Adaptation** |    arXiv   | [Link](https://arxiv.org/pdf/2406.11909) |  [Link](https://github.com/wutaiqiang/MoSLoRA) | Mixtureof-Subspaces; Robust Fine-Tuning |
| 2024 | **ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts** |    arXiv   | [Link](https://arxiv.org/pdf/2406.10973) |  - | VIT; Unsupervised Pre-Training; Supervised Learning |
| 2024 | **ShareLoRA: Parameter Efficient and Robust Large Language Model Fine-tuning via Shared Low-Rank Adaptation** |    arXiv   | [Link](https://arxiv.org/pdf/2406.10785) |  - | Shared; transfer learning |
| 2024 | **ALoRA: Allocating Low-Rank Adaptation for Fine-tuning Large Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2403.16187) |  - | Allocating; Structural Pruning |
| 2024 | **ResLoRA: Identity Residual Mapping in Low-Rank Adaption** |    arXiv   | [Link](https://arxiv.org/pdf/2402.18039) |  [Link](https://github.com/microsoft/LMOps/tree/main/reslora) | Residual Paths |
| 2024 | **RST-LoRA: A Discourse-Aware Low-Rank Adaptation for Long Document Abstractive Summarization** |    arXiv   | [Link](https://arxiv.org/pdf/2405.00657) |  - | Rhetorical Structure Theory (RST); Long Document |
| 2024 | **Federated LoRA with Sparse Communication** |    arXiv   | [Link](https://arxiv.org/pdf/2406.05233) | [Link](https://github.com/imkevinkuo/flasc) | Communication-Efficiency in Federated LoRA |
| 2024 | **RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation of Pre-trained Language Model for Knowledge Editing and Fine-tuning** |    arXiv   | [Link](https://arxiv.org/pdf/2406.10777) |  - | The Sparsity with Respective to The Matrix Product |
| 2024 | **Task-Aware Low-Rank Adaptation of Segment Anything Model** |    arXiv   | [Link](https://arxiv.org/pdf/2403.10971) |  - | Segment Anything Model (SAM); Multi-Task Learning
| 2024 | **Relora: High-rank training through low-rank updates** |    ICLR   | [Link](https://arxiv.org/pdf/2307.05695) | [Link](https://github.com/Guitaricet/relora) | Low-Rank Updates |
| 2024 | **Low-Rank Few-Shot Adaptation of Vision-Language Models** |    CVPR   | [Link](https://arxiv.org/pdf/2405.18541) |  [Link](https://github.com/MaxZanella/CLIP-LoRA) | VisionLanguage Models (VLMs); Few-Shot |
| 2024 | **MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning** |    CVPR   | [Link](https://arxiv.org/pdf/2403.20320) | [Link](https://github.com/scale-lab/MTLoRA) | Multi-Task Learning (MTL); Pareto-Optimal Trade-Off |
| 2024 | **QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2309.14717) |  [Link](https://github.com/eltociear/qa-lora) | Quantization and Adaptation; Group-Wise Operators |
| 2024 | **Defending Against Weight-Poisoning Backdoor Attacks for Parameter-Efficient Fine-Tuning** |    arXiv   | [Link](https://arxiv.org/pdf/2402.12168) |  - | Security Vulnerabilities; Poisoned Sample Identification Module (PSIM) |
| 2024 | **Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models** |    COLING   | [Link](https://arxiv.org/pdf/2403.03432) |  - | Mixture-of-LoRAs; Robust Fine-Tuning |
| 2024 | **LaMDA: Large Model Fine-Tuning via Spectrally Decomposed Low-Dimensional Adaptation** |    arXiv   | [Link](https://arxiv.org/pdf/2406.12832) |  [Link](https://github.com/ArminAzizi98/LaMDA) | Projection Matrix (PM); Lite-Weight | 
| 2024 | **Accurate LoRA-Finetuning Quantization of LLMs via Information Retention** |    ICML   | [Link](https://arxiv.org/pdf/2402.05445) | [Link](https://github.com/htqin/IR-QLoRA) | Quantization; Information Retention;|
| 2024 | **Quantum-informed Tensor Adaptation (QuanTA): Efficient High-Rank Fine-Tuning of Large Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2406.00132) | [Link](https://github.com/quanta-fine-tuning/quanta) | Quantum-informed Tensor Adaptation (QuanTA) |
| 2024 | **VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks** |    arXiv   | [Link](https://arxiv.org/pdf/2405.15179) |  [Link](https://github.com/leo-yangli/VB-LoRA) | Shared Vector Bank |
| 2024 | **MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning** |    arXiv   | [Link](https://arxiv.org/pdf/2405.12130) |  [Link](https://github.com/kongds/MoRA) | High-Rank Updating; Non-Parameter Operators |
| 2024 | **FLoRA: Low-Rank Core Space for N-dimension** |    arXiv   | [Link](https://arxiv.org/pdf/2405.14739) |  [Link](https://github.com/SJTU-DeepVisionLab/FLoRA) | N-Dimensional Parameter Space |
| 2024 | **LOFIT: Localized Fine-tuning on LLM Representations** |    arXiv   | [Link](https://arxiv.org/pdf/2406.01563) |  [Link](https://github.com/fc2869/lo-fit) | Localized Fine-Tuning |
| 2024 | **Visual Perception by Large Language Model's Weights** |    arXiv   | [Link](https://arxiv.org/pdf/2405.20339) |  - | Visual Perception |
| 2024 | **Memory-Space Visual Prompting for Efficient Vision-Language Fine-Tuning** |    ICML   | [Link](https://arxiv.org/pdf/2405.05615) |  [Link](https//github.com/JieShibo/MemVP) | Vision-Lnguage (VL); Memoryspace Visual Prompting (MemVP) |
| 2024 | **AdvLoRA: Adversarial Low-Rank Adaptation of Vision-Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2404.13425) |  - |Robust Fine-Tuning; Adversarial Robustness; Vision-Language Models; Clustering;|
| 2024 | **Parameter-Efficient Fine-Tuning with Discrete Fourier Transform** |    ICML   | [Link](https://arxiv.org/pdf/2405.03003) |  [Link](https://github.com/Chaos96/fourierft) | Discrete Fourier Transform |
| 2024 | **LoNAS: Elastic Low-Rank Adapters for Efficient Large Language** |    COLING   | [Link](https://aclanthology.org/2024.lrec-main.940.pdf) |  [Link](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning?tab=readme-ov-file) | Neural Architecture Search； Parameter-Efficient Fine-Tuning |
| 2024 | **LoRA Learns Less and Forgets Less** |    arXiv   | [Link](https://arxiv.org/pdf/2405.09673) |  - | Robust Fine-Tuning; Adversarial Robustness |
| 2024 | **LoRA+: Efficient Low Rank Adaptation of Large Models** |    arXiv   | [Link](https://arxiv.org/abs/2402.12354) |  [Link](https://github.com/nikhil-ghosh-berkeley/loraplus) | Efficient Fine-Tuning |
| 2024 | **PeriodicLoRA: Breaking the Low-Rank Bottleneck in LoRA Optimization** |    arXiv   | [Link](https://arxiv.org/abs/2402.16141) |  - | Low-Rank Bottleneck |
| 2024 | **Sparse Matrix in Large Language Model Fine-tuning** |    arXiv   | [Link](https://arxiv.org/html/2405.15525v1) |  - | Sparse Matrix Tuning (SMT); Robust Fine-Tuning |
| 2024 | **Derivative-Free Optimization for Low-Rank Adaptation in Large Language Models** |    arXiv   | [Link](https://arxiv.org/abs/2403.01754) |  [Link](https://github.com/stan-anony/derivative_free_lora_rank) | Derivative-Free Optimization; Robust Fine-Tuning |
| 2024 | **Multi-LoRA Composition for Image Generation** |    arXiv   | [Link](https://arxiv.org/pdf/2402.16843) |  [Link](https://github.com/maszhongming/Multi-LoRA-Composition) | Multi-LoRA Composition; Text-to-Image Models |
| 2024 | **BiLoRA: A Bi-level Optimization Framework for Overfitting-Resilient Low-Rank Adaptation of Large Pre-trained Models** |    arXiv   | [Link](https://arxiv.org/abs/2403.13037) |  - | Bi-Level Optimization (BLO); Robust Fine-Tuning |
| 2024 | **AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter Efficient Fine-Tuning of Large Models** |    arXiv   | [Link](https://arxiv.org/abs/2403.13269) |  - | Adaptive Freezing; Robust Fine-Tuning |
| 2024 | **LoRA Meets Dropout under a Unified Framework** |    arXiv   | [Link](https://arxiv.org/abs/2403.00812) |  - | HiddenKey; Dropout; Robust Fine-Tuning |
| 2024 | **Galore: Memory-efficient llm training by gradient low-rank projection** |    ICML   | [Link](https://arxiv.org/abs/2403.03507) |  [Link](https://github.com/jiaweizzhao/GaLore) | Gradient Low-Rank Projection (GaLore)；Robust Fine-Tuning |
| 2024 | **Let's Focus on Neuron: Neuron-Level Supervised Fine-tuning for Large Language Model** |    arXiv   | [Link](https://arxiv.org/abs/2403.11621) |  - | Neuron-Level Fine-Tuning (NeFT); Robust Fine-Tuning |
| 2024 | **LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning** |    arXiv   | [Link](https://arxiv.org/abs/2403.17919) |  - | Layerwise Importance Sampled AdamW (LISA); Robust Fine-Tuning |
| 2023 | **Efficient Low-rank Backpropagation for Vision Transformer Adaptation** |    NeurIPS   | [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/2f75a57e9c71e8369da0150ea769d5a2-Paper-Conference.pdf) |  [Link](https://github.com/SLDGroup/LBP-WHT) | vision transformers (ViT); Robust Fine-Tuning |
| 2023 | **Delta-LoRA: Fine-Tuning High-Rank Parameters with the Delta of Low-Rank Matrices** |    arXiv   | [Link](https://arxiv.org/pdf/2309.02411) |  - | Robust Fine-Tuning; Adversarial Robustness |
| 2023 | **DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation** |    EACL   | [Link](https://arxiv.org/pdf/2210.07558) |  [Link](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA) | SVD Modules; Pretrained Models (PMs); Robust Fine-Tuning |
| 2023 | **The expressive power of low-rank adaptation** |    ICLR   | [Link](https://arxiv.org/abs/2310.17513) |  [Link](https://github.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA) | THE EXPRESSIVE POWER |
| 2023 | **Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of RLHF** |    arXiv   | [Link](https://arxiv.org/abs/2309.09055) |  [Link](https://github.com/SimengSun/alpaca_farm_lora) | RLHF; Robust Fine-Tuning |
| 2023 | **Deep Learning Model Compression With Rank Reduction in Tensor Decomposition** |    TNNLS   | [Link](https://ieeexplore.ieee.org/abstract/document/10321737) |  - | Rank Reduction in Tensor Decomposition; Robust Fine-Tuning |
| 2023 | **Loramoe: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment** |    arXiv   | [Link](https://simg.baai.ac.cn/paperfile/96f0cfd7-79c7-4110-88e5-4ea80a7fbc8d.pdf) |  - | Supervised fine-tuning (SFT); Mixture of Experts (MoE); Robust Fine-Tuning |
| 2023 | **Bayesian Low-rank Adaptation for Large Language Models** |    ICLR   | [Link](https://arxiv.org/abs/2308.13111) |  [Link](https://github.com/MaximeRobeyns/bayesian_lora) | Laplace approximation; Robust Fine-Tuning |
| 2023 | **Lora-fa: Memory-efficient low-rank adaptation for large language models fine-tuning** |    arXiv   | [Link](https://arxiv.org/abs/2308.03303) |  - | Memory of Large Language Models; Robust Fine-Tuning |
| 2023 | **Motion Style Transfer: Modular Low-Rank Adaptation for Deep Motion Forecasting** |    PMLR   | [Link](https://proceedings.mlr.press/v205/kothari23a/kothari23a.pdf) |  [Link](https://github.com/vita-epfl/motion-style-transfer) | Motion Forecasting; Distribution Shifts; Transfer Learning |
| 2023 | **Sparse low-rank adaptation of pre-trained language models** |    EMNLP   | [Link](https://aclanthology.org/2023.emnlp-main.252.pdf) |  [Link](https://github.com/TsinghuaC3I/SoRA) | Sparse Low-Rank; Robust Fine-Tuning |
| 2023 | **Low-Rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition** |    ASRU   | [Link](https://ieeexplore.ieee.org/abstract/document/10389632) |  - | Parameter-Efficient Speech Recognition |
| 2023 | **SiRA: Sparse Mixture of Low Rank Adaptation** |    arXiv   | [Link](https://arxiv.org/abs/2311.09179) |  - | Sparse Mixture of Expert(SMoE); Robust Fine-Tuning |
| 2021 | **Compacter: Efficient low-rank hypercomplex adapter layers** |    NeurIPS   | [Link](https://proceedings.neurips.cc/paper/2021/file/081be9fdff07f3bc808f935906ef70c0-Paper.pdf) |  [Link](https://github.com/rabeehk/compacter) |
| 2022 | **LoRA: Low-Rank Adaptation of Large Language Models** |    ICLR   | [Link](https://arxiv.org/abs/2106.09685) |  [Link](https://github.com/microsoft/LoRA) | The Pioneering Paper |


## Packages
Huggingface PEFT [Link](https://github.com/huggingface/peft) 




