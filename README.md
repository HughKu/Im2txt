# Note:
This repo aims to provide a **Ready-to-Go** setup with TensorFlow environment for **Image Captioning Inference** using pre-trained model. For training from scratch or funetuning, please refer to [Tensorflow Model Repo](https://github.com/tensorflow/models/tree/master/research/im2txt).


# Contents
* [Model Overview](#model-overview)
    * [Introduction](#introduction)
    * [Architecture](#architecture)
* [Requirement](#getting-started)
    * [Install](#install-required-packages)
    * [Get Pre-trained Model](#get-pre-trained-model)
* [Generating Captions](#generating-captions)
* [Issue](#encoutering-issue)

## Model Overview

### Introduction
The *Show and Tell* model is a deep neural network that learns how to describe
the content of images. For example:

![Example captions](g3doc/example_captions.jpg)

*Show and Tell: A Neural Image Caption Generator*

A TensorFlow implementation of the image-to-text model described in the paper:

"Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning
Challenge."

Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan.

*IEEE transactions on pattern analysis and machine intelligence (2016).*

Full text available at: http://arxiv.org/abs/1609.06647

### Architecture
Please refer to the original [Tensorflow Model Repo](https://github.com/tensorflow/models/tree/master/research/im2txt).

## Requirement

### Install Required Packages
I strongly suggest that you run `pip install -r requirement.txt` in your CLI 
to get all packages needed.

OR you could opt for manually installing the required packages below:

* **TensorFlow** 1.0 or greater ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](http://www.scipy.org/install.html))
* **Natural Language Toolkit (NLTK)**:
    * First install NLTK ([instructions](http://www.nltk.org/install.html))
    * Then install the NLTK data package "punkt" ([instructions](http://www.nltk.org/data.html))

### Get Pre-trained Model
Download [inceptionv3 finetuned parameters over 1M](https://drive.google.com/open?id=1r4-9FEIbOUyBSvA-fFVFgvhFpgee6sF5) and you will get 4 files, and make sure to put them all into this path `im2txt/model/Hugh/train/`
* **newmodel.ckpt-2000000.data-00000-of-00001**
* **newmodel.ckpt-2000000.index**
* **newmodel.ckpt-2000000.meta**
* **checkpoint**

## Generating Captions
Your downloaded *Show and Tell* model can generate captions for any JPEG image! The
following command line will generate captions for such an image.
```
python im2txt/run_inference.py --checkpoint_path="im2txt/model/Hugh/train/newmodel.ckpt-2000000" --vocab_file="im2txt/data
/Hugh/word_counts.txt" --input_files="im2txt/data/images/test.jpg"
```

Example output:
```
Captions for image test.jpg:
  0) a young boy wearing a hat and tie . (p=0.000195)
  1) a young boy wearing a blue shirt and tie . (p=0.000100)
  2) a young boy wearing a blue shirt and a tie . (p=0.000045)
```

Note: you may get different results. Some variation between different models is
expected.

Here is the image:

![ME](im2txt/data/images/test.jpg)

## Encoutering Issue
First, check out on this [thread](https://github.com/tensorflow/models/issues/466) and it's likely that you find answer there. Otherwise, open an issue and I will try to help you.
