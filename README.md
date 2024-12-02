# Reproduction Report: Temporal difference for credit assignment and exploration of TD(λ) family of learning procedures
### Introduction
Reinforcement learning is a major branch of machine learning which uses sequence of past experiences, to help a system learn and predict optimal behaviour. There are multitude of proposed learning procedures that have attempted to perform such learning. While typical prediction problems utilized the final outcome to minimize the error, Sutton in \cite{sutton1988learning} explores "temporally successive predictions" to assign credit for the actions performed. Similar to a weather forecast for the weekend getting better as time progresses, TD (Temporal Difference) learning utilizes the same method of updating estimates every time step to progress towards the optimal value. This technical report discusses the background and intuition behind the original paper  \cite{sutton1988learning} by recreating the experiments and investigates the reproduced empirical results to compare and contrast the assumptions and findings described in the original literature.

### Citation
If you find this project useful in your research or work, please consider citing it:
```
@article{gopal2024tdrepro,
  title={Reproduction Report: Temporal difference for credit assignment and exploration of TD(λ) family of learning procedures},
  author={Gopal, Senthilkumar},
  journal={arXiv preprint arXiv:tbd},
  year={2024}
}
```
Following are the files and folders and their usage.

## Files

### environment.yml
1. This file contains the Conda environment configuration.
2. Use `conda env create -f environment. yml` to create the `repro` environment
3. Activate the new environment: `conda activate repro`
