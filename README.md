# CS242 Final Project

## Overview

RNNs are a neural network architecture for language modelling, audio transcription, translation, and other tasks. While much work has been put into accelerating the training of _other_ architectures like CNNs, not much progress has been made on RNNs. Parallelizing and training RNNs is inherently difficult because it must sequentially process every token of a sentence. In this project, we propose and develop a novel way of training RNNs, which has similar convergence properties to SGD, but still maintains high data parallelism and hardware utilization in theory.

## Installation

* We assume the user will be using `python3`.
* Install the requisite packages with `pip install -r requirements.txt`

## Running

```bash
python main.py
```
