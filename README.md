# DynamicTriad-pytorch
This repository aims to implement and reproduce **DynamicTriad** using PyTorch. 
**DynamicTriad** has been proposed in the paper "Dynamic network embedding by modeling triadic closure process (AAAI 2018)", and its original implementation is [here](https://github.com/luckiezhou/DynamicTriad) implemented in TensorFlow. 

## Datasets
We use the following datasets to reproduce the experimental results shown in the paper. 

* Academic
* Academic_toy

The Academic and Academic_toy datasets have been obtained by running [academic2adjlist.py](https://github.com/luckiezhou/DynamicTriad/blob/master/scripts/academic2adjlist.py)
in the original repository.

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


### Batch duplication
In this repository, we introduce `batch duplication` to boost up the speed of the training procedure. 
The main idea of the batch duplication is to use sampled data multiple times to produce batch data without repeating the sampling for each step. 
For example, the pseudocode of the original training procedure is as follows:

```python
for epoch in range(epochs):
    sample = gen_sample()
    for batch in gen_batch(sample):
        model.train(batch)
```
In the above code, `gen_sample()` is time-consuming, thereby slowing down the training phase overall. 
To accelerate this phase, the batch duplicated code is written as follows:

```python
for epoch in range(epochs):
    sample = gen_sample()
    for _ in range(batdup):   # batch duplication
        for batch in gen_batch(sample):
            model.train(batch)
```
Note that `sample` is reused `batdup` times, which is controlled by a user as a hyperparameter. 
With a reduced `epochs`, the batch duplication can decrease the training time.
(Of course, there is a trade-off between efficiency and accuracy because the batch duplication could harm the randomness of the sampling, but its effect seems scant as shown in the below). 
If `batdup` is set to `1`, then the batch duplicated version is the same as the original one.

## Evaluation results

We have tested this repository on the following tasks compared to the original results. 
We report average accuracies with their standard deviations of 10 runs. 

| **Mode** | **Original (paper)** | **This** |
| :--- | :---: | :---: |
| **Link reconstruction** | 0.985 | 0.958±0.0002 |
| **Link prediction** | 0.836 | 0.949±0.0002 |

## References
[1] Zhou, L., Yang, Y., Ren, X., Wu, F., & Zhuang, Y. (2018, April). Dynamic network embedding by modeling triadic closure process. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 32, No. 1).
