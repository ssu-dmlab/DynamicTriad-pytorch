# DynamicTriad-pytorch
This repository aims to implement and reproduce **DynamicTriad** using PyTorch. 
**DynamicTriad** has been proposed in the paper "Dynamic Network Embedding by Modelling Triadic Closure Process (AAAI 2018)", and its original implementation is [here](https://github.com/luckiezhou/DynamicTriad) implemented in TensorFlow. 

## Datasets
We use the following datasets to reproduce the experimental results shown in the paper. 

* Academic
* Academic_toy

The Academic and Academic_toy datasets have been obtained by running [academic2adjlist.py](https://github.com/luckiezhou/DynamicTriad/blob/master/scripts/academic2adjlist.py)
in the original repository.

## Differences between the original implementation and this
We summarize the differences between the original repository and this. 
We mainly focus on writing pythonic codes for the method using PyTorch.

| **Item** | **Original** | **This** |
| :--- | :--- | :--- |
| Python version | 2 | 3 |
| ML library | TensorFlow | PyTorch |
| Graph implementation | C++ | Python |
| Multiprocessing for sampling | Yes | No |
| Batch duplication | No | Yes |
| Vertex label | Yes | No |

## Usage

You can run this project to simply type the following in your terminal:

```shell
python -m src.main \
	--model=original \
	--dir=${DIR} \
	--dataset=${DATASET_NAME} \
	--device=${cpu/cuda} \
	--epochs=... \
	--lr=... \
	--time_length=... \
	--time_step=... \
	--time_stride=... \
	--emb_dim=... \
	--beta_triad=... \
	--beta_smooth=... \
	--batchsize=... \
	--batdup=... \
	--mode=...
```

| **Option** | **Description** | **Default** |
|:--- | :--- | :---: |
| `model` | Which model to use (only `original` is implemented currently) | `original` |
| `dir` | Directory where dataset is located | `datasets`|
|`dataset`| Dataset to train (`academic` or `academic_toy`) | `academic`|
| `device` | Torch device to use (`cpu` or `cuda`) | `cpu`|
| `epochs` | Number of training epochs | 300 |
| `lr` | Learning rate | 0.1 |
| `time_length` | Time length to load from raw dataset | 36 |
| `time_step` | Time step to merge from raw dataset | 4 |
| `time_stride` | Time stride to jump time when merge from raw dataset | 2 |
| `emb_dim` | Embedding dimension | 48 |
| `beta_triad` | Hyperparameter for triad loss | 1.0 |
| `beta_smooth` | Hyperparameter for smoothness loss | 1.0 |
| `batchsize` | Batch size | 10000 |
| `batdup` | Batch duplication, hyperparameter to reuse same sample | 5 |
| `mode` | Evaluation mode: `link_{reconstruction,prediction}` | `link_reconstruction` |

## Merge example

If time length is 16, time step is 4, time stride is 2, dataset is merged as below:

```
 0, 1, 2, 3 -> 0
 2, 3, 4, 5 -> 1
 4, 5, 6, 7 -> 2
 6, 7, 8, 9 -> 3
 8, 9,10,11 -> 4
10,11,12,13 -> 5
12,13,14,15 -> 6
```

## Batch duplication example

Original training pseudocode is:
```python
for epoch in range(epochs):
	sample = gen_sample()
	for batch in gen_batch(sample):
		model.train(batch)
```

Training pseudocode with batch duplication is:
```python
for epoch in range(epochs):
	sample = gen_sample()
	for _ in range(batdup):
		for batch in gen_batch(sample):
			model.train(batch)
```

You can reuse sample by using batdup hyperparameter. This is implemented due to slow sampling process.

## Evaluation results

We have tested this repository on the following tasks compared to the original results. 
We report average accuracies with their standard deviations of 10 runs. 

| **Mode** | **Original (paper)** | **This** |
| :--- | :---: | :---: |
| **Link reconstruction** | 0.985 | 0.958±0.0002 |
| **Link prediction** | 0.836 | 0.949±0.0002 |
