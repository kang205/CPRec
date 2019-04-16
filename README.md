# CPRec: Learning Consumer and Producer Embeddings for User-Generated Content Recommendation

This is our TensorFlow implementation for the paper:

Wang-Cheng Kang, Julian McAuley. *[Learning Consumer and Producer Embeddings for User-Generated Content Recommendation.](https://arxiv.org/pdf/1809.09739.pdf)* In Proceedings of ACM Conference on Recommender System (RecSys'18)

Please cite our paper if you use the code or datasets.

The code is tested under a Linux desktop with TensorFlow 1.12.

## Datasets

We describe how to obtain the raw data of `Reddit` and `Pinterest` in the `data` folder, and data processing scripts are also included. The `Reddit` data (after pre-processing) is also available.

## Model Training

A quick way to train our model is (with default hyper-parameters): 

```
python main.py --dataset=RedditCore 
```

In 100 epochs, you should be able to see the test AUC in the log file reach 0.9. With more epochs, it can be further improved.

For more details (e.g. learning rate, regularizations, etc), please refer to the code. 
