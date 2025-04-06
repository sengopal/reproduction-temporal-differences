# Reproduction Report: Temporal difference for credit assignment and exploration of TD(λ) family of learning procedures
### Introduction
Reinforcement learning is a major branch of machine learning which uses sequence of past experiences, to help a system learn and predict optimal behaviour. There are multitude of proposed learning procedures that have attempted to perform such learning. While typical prediction problems utilized the final outcome to minimize the error, Sutton explores "temporally successive predictions" to assign credit for the actions performed. Similar to a weather forecast for the weekend getting better as time progresses, TD (Temporal Difference) learning utilizes the same method of updating estimates every time step to progress towards the optimal value. This technical report discusses the background and intuition behind the original paper (https://incompleteideas.net/papers/sutton-88-with-erratum.pdf) by recreating the experiments and investigates the reproduced empirical results to compare and contrast the assumptions and findings described in the original literature.

### Citation
If you find this project useful in your research or work, please consider citing it:
```
@article{tempsenthilrepro,
	doi = {10.20944/preprints202412.0430.v1},
	url = {https://doi.org/10.20944/preprints202412.0430.v1},
	year = 2024,
	month = {December},
	publisher = {Preprints},
	author = {Senthilkumar Gopal},
	title = {Temporal Difference for Credit Assignment and Exploration of TD(λ) Family of Learning Procedures – A Reproducibility Report},
	journal = {Preprints}
}
```
Following are the files and folders and their usage.

## Files

### environment.yml
1. This file contains the Conda environment configuration.
2. Use `conda env create -f environment. yml` to create the `repro` environment
3. Activate the new environment: `conda activate repro`
