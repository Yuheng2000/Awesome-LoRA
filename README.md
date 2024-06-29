# Awesome-LoRA
[python-img]: https://img.shields.io/github/languages/top/yuheng2000/Awesome-LoRA?color=lightgrey
[stars-img]: https://img.shields.io/github/stars/yuheng2000/Awesome-LoRA?color=yellow
[stars-url]: https://github.com/yuheng2000/Awesome-LoRA/stargazers
[fork-img]: https://img.shields.io/github/forks/yuheng2000/Awesome-LoRA?color=lightblue&label=fork
[fork-url]: https://github.com/yuheng2000/Awesome-LoRA/network/members
[visitors-img]: https://badges.pufler.dev/visits/yuheng2000/Awesome-LoRA
[adgc-url]: https://github.com/yuheng2000/Awesome-LoRA



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
| - | **-** |    -   | - |  - |



## Papers

| Year | Title                                                        | **Venue** |                            Paper                             |                             Code                             |
| :--: | :----------------------------------------------------------- | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024 | **Defending Against Weight-Poisoning Backdoor Attacks for Parameter-Efficient Fine-Tuning** |    arXiv   | [Link](https://arxiv.org/pdf/2402.12168) |  - |
| 2024 | **Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models** |    COLING   | [Link](https://arxiv.org/pdf/2403.03432) |  - |
| 2024 | **LaMDA: Large Model Fine-Tuning via Spectrally Decomposed Low-Dimensional Adaptation** |    arXiv   | [Link](https://arxiv.org/pdf/2406.12832) |  - |
| 2024 | **Accurate LoRA-Finetuning Quantization of LLMs via Information Retention** |    arXiv   | [Link](https://arxiv.org/pdf/2402.05445) | [Link](https://github.com/htqin/IR-QLoRA) |
| 2024 | **ResLoRA: Identity Residual Mapping in Low-Rank Adaption** |    arXiv   | [Link](https://arxiv.org/pdf/2402.18039) |  - |
| 2024 | **Quantum-informed Tensor Adaptation (QuanTA): Efficient High-Rank Fine-Tuning of Large Language Models** |    arXiv   | [Link](https://arxiv.org/abs/2406.00132) | [Link](https://github.com/quanta-fine-tuning/quanta) |
| 2024 | **VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks** |    arXiv   | [Link](https://arxiv.org/pdf/2405.15179) |  - |
| 2024 | **MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning** |    arXiv   | [Link](https://arxiv.org/pdf/2405.12130) |  - |
| 2024 | **FLoRA: Low-Rank Core Space for N-dimension** |    arXiv   | [Link](https://arxiv.org/pdf/2405.14739) |  [Link](https://github.com/SJTU-DeepVisionLab/FLoRA) |
| 2024 | **LOFIT: Localized Fine-tuning on LLM Representations** |    arXiv   | [Link](https://arxiv.org/pdf/2406.01563) |  - |
| 2024 | **Visual Perception by Large Language Model's Weights** |    arXiv   | [Link](https://arxiv.org/pdf/2405.203) |  - |
| 2024 | **Memory-Space Visual Prompting for Efficient Vision-Language Fine-Tuning** |    ICML   | [Link](https://arxiv.org/pdf/2405.05615) |  [Link](https//github.com/JieShibo/MemVP) |
| 2024 | **Compressible Dynamics in Deep Overparameterized Low-Rank Learning & Adaptation** |    ICML   | [Link](https://arxiv.org/pdf/2406.04112) |  - |
| 2024 | **AdvLoRA: Adversarial Low-Rank Adaptation of Vision-Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2404.13425) |  - |
| 2024 | **FLoRA: Low-Rank Core Space for N-dimension** |    arXiv   | [Link](https://arxiv.org/abs/2405.14739) |  [Link](https://github.com/SJTU-DeepVisionLab/FLoRA) |
| 2024 | **Parameter-Efficient Fine-Tuning with Discrete Fourier Transform** |    ICML   | [Link](https://arxiv.org/pdf/2405.03003) |  [Link](https://github.com/Chaos96/fourierft) |
| 2024 | **LoNAS: Elastic Low-Rank Adapters for Efficient Large Language** |    COLING   | [Link](https://aclanthology.org/2024.lrec-main.940.pdf) |  [Link](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning?tab=readme-ov-file) |
| 2024 | **LoRA Learns Less and Forgets Less** |    arXiv   | [Link](https://arxiv.org/pdf/2405.09673) |  - |
| 2024 | **MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning** |    arXiv   | [Link](https://arxiv.org/pdf/2405.12130) |  [Link](https://github.com/kongds/MoRA) |
| 2024 | **LoRA+: Efficient Low Rank Adaptation of Large Models** |    arXiv   | [Link](https://arxiv.org/abs/2402.12354) |  [Link](https://github.com/nikhil-ghosh-berkeley/loraplus) |
| 2024 | **PeriodicLoRA: Breaking the Low-Rank Bottleneck in LoRA Optimization** |    arXiv   | [Link](https://arxiv.org/abs/2402.16141) |  - |
| 2024 | **Sparse Matrix in Large Language Model Fine-tuning** |    arXiv   | [Link](https://arxiv.org/html/2405.15525v1) |  - |
| 2024 | **Derivative-Free Optimization for Low-Rank Adaptation in Large Language Models** |    arXiv   | [Link](https://arxiv.org/abs/2403.01754) |  [Link](https://github.com/stan-anony/derivative_free_lora_rank) |
| 2024 | **Multi-LoRA Composition for Image Generation** |    arXiv   | [Link](https://arxiv.org/pdf/2402.16843) |  [Link](https://github.com/maszhongming/Multi-LoRA-Composition) |
| 2024 | **BiLoRA: A Bi-level Optimization Framework for Overfitting-Resilient Low-Rank Adaptation of Large Pre-trained Models** |    arXiv   | [Link](https://arxiv.org/abs/2403.13037) |  - |
| 2024 | **AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter Efficient Fine-Tuning of Large Models** |    arXiv   | [Link](https://arxiv.org/abs/2403.13269) |  - |
| 2024 | **LoRA Meets Dropout under a Unified Framework** |    arXiv   | [Link](https://arxiv.org/abs/2403.00812) |  - |
| 2024 | **MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning** |    arXiv   | [Link](https://arxiv.org/pdf/2403.20320) |  [Link](https://github.com/scale-lab/MTLoRA) |
| 2024 | **Galore: Memory-efficient llm training by gradient low-rank projection** |    ICML   | [Link](https://arxiv.org/abs/2403.03507) |  [Link](https://github.com/jiaweizzhao/GaLore) |
| 2024 | **Let's Focus on Neuron: Neuron-Level Supervised Fine-tuning for Large Language Model** |    arXiv   | [Link](https://arxiv.org/abs/2403.11621) |  - |
| 2024 | **LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning** |    arXiv   | [Link](https://arxiv.org/abs/2403.17919) |  - |
| 2023 | **DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation** |    EACL   | [Link](https://arxiv.org/pdf/2404.13425) |  [Link](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA) |
| 2023 | **The expressive power of low-rank adaptation** |    ICLR   | [Link](https://arxiv.org/abs/2310.17513) |  [Link](https://github.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA) |
| 2023 | **Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of RLHF** |    arXiv   | [Link](https://arxiv.org/abs/2309.09055) |  [Link](https://github.com/SimengSun/alpaca_farm_lora) |
| 2023 | **Deep Learning Model Compression With Rank Reduction in Tensor Decomposition** |    TNNLS   | [Link](https://ieeexplore.ieee.org/abstract/document/10321737) |  - |
| 2023 | **Loramoe: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment** |    arXiv   | [Link](https://simg.baai.ac.cn/paperfile/96f0cfd7-79c7-4110-88e5-4ea80a7fbc8d.pdf) |  - |
| 2023 | **Bayesian Low-rank Adaptation for Large Language Models** |    ICLR   | [Link](https://arxiv.org/abs/2308.13111) |  [Link](https://github.com/MaximeRobeyns/bayesian_lora) |
| 2023 | **Lora-fa: Memory-efficient low-rank adaptation for large language models fine-tuning** |    arXiv   | [Link](https://arxiv.org/abs/2308.03303) |  - |
| 2023 | **Motion Style Transfer: Modular Low-Rank Adaptation for Deep Motion Forecasting** |    PMLR   | [Link](https://proceedings.mlr.press/v205/kothari23a/kothari23a.pdf) |  [Link](https://github.com/vita-epfl/motion-style-transfer) |
| 2023 | **Sparse low-rank adaptation of pre-trained language models** |    EMNLP   | [Link](https://aclanthology.org/2023.emnlp-main.252.pdf) |  [Link](https://github.com/TsinghuaC3I/SoRA) |
| 2023 | **Low-Rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition** |    ASRU   | [Link](https://ieeexplore.ieee.org/abstract/document/10389632) |  - |
| 2023 | **SiRA: Sparse Mixture of Low Rank Adaptation** |    arXiv   | [Link](https://arxiv.org/abs/2311.09179) |  - |
| 2022 | **LoRA: Low-Rank Adaptation of Large Language Models** |    ICLR   | [Link](https://arxiv.org/abs/2106.09685) |  [Link](https://github.com/microsoft/LoRA) |

## Others
| Year | Title                                                        | **Venue** |                            Paper                             |                             Code                             |
| :--: | :----------------------------------------------------------- | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2021 | **Compacter: Efficient low-rank hypercomplex adapter layers** |    NeurIPS   | [Link](https://proceedings.neurips.cc/paper/2021/file/081be9fdff07f3bc808f935906ef70c0-Paper.pdf) |  [Link](https://github.com/rabeehk/compacter) |



## Packages
Huggingface PEFT [Link](https://github.com/huggingface/peft) 




