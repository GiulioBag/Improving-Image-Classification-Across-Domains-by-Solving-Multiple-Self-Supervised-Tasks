# Improving Image Classification Across Domains by Solving Multiple Self-Supervised Tasks

Base code for performing experiments with a multi-task-selfsupervised network.

## Abstract
One thing all machine learning scholars agree on is that a secret to getting a model that works well is that we need to train it on a huge dataset. Some of them are extremely expensive to collect and it is not always possible to recover a large amount of data. Aware of this problem, two solutions have been proposed: Domain Shift, which consists in to train the model on data distributions similar to those on which it will operate, and Self Supervised Learning, which is a paradigm that attempts to get the machines to derive supervision patterns from the data itself, without human involvement. The proposed approach combines these two strategies by building a model capable of learning features independent of the domain from which the images come. This knowledge is then applied to the dataset on which the model will operate. To extrapolate these features various techniques have been proposed that have been tested individually and in groups, reporting significant improvements compared to the simple use of Domain Shift strategies. Contrary to the standard approach of Self Supervised Learning, in our idea, the network trains on the main task and on the secondary ones in parallel. Both contribute to the model's weights update.

## Dataset
The experiments must be conducted on the PACS database.

1 - Download PACS dataset from here http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017

2 - Put the database in the "Improving-Image-Classification-Across-Domains-by-Solving-Multiple-Self-Supervised-Tasks" folder

## Pretrained model
For the DG experiments, in order to reproduce exactly the values reported in the tables, you have to use the "caffe" pretrained model.

You can download it from here: https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing

Then, you have to put it into 

```
/Improving-Image-Classification-Across-Domains-by-Solving-Multiple-Self-Supervised-Tasks/models/pretrained/
```

## Requirements
To run the experiments you have to install all the required libraries listed in the "requirements.txt" file.
