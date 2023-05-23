# STAN: Spatial-Temporal-Attention-Network-for-Next-Location-Recommendation
[[Paper](https://arxiv.org/abs/2102.04095)]. [[Oral Youtube](https://www.youtube.com/watch?v=ajNzESvOvzs)] or [[Oral Bilibili](https://www.bilibili.com/video/BV1WL411E7Qm?from=search&seid=7472683569881802215)]. [[Implementation through LibCity](https://github.com/LibCity/Bigscity-LibCity)].

Thank you for your interest in our work! Thank you for reporting possible bugs and please make sure you are forking the latest repo to avoid eariler bugs. Before asking questions regarding the codes or the paper, I strongly recommend you to read the FAQ first. You can also use the [LibCity](https://github.com/LibCity/Bigscity-LibCity) version.

## Description
Because of the huge memory of the location matrix, the running speed of STAN is extremely low. You can refer to the implementation of masked attention [[here](https://github.com/yingtaoluo/PyHealth/blob/master/pyhealth/models/sequence/dipole.py)] if you wish to rewrite your own codes. 

Divide the dataset into different proportions of users to test the performance and then average. 

Run "load.py" first and then "train.py". You should see on the screen the result of the first proportion:   
100%|██████████| 100/100 [14:32<00:00,  8.72s/it]  
epoch:27, time:23587.941201210022, valid_acc:[0.18 0.49 0.56 0.67]  
epoch:27, time:23587.941201210022, test_acc:[0.15 0.46 0.59 0.67]

## FAQs
Q1: Can you provide a dataset?  
A1: Our datasets are collected from the following links. Please feel free to do your own data processing on your model while comparing STAN as baseline.
http://snap.stanford.edu/data/loc-gowalla.html;  
https://personal.ntu.edu.sg/gaocong/data/poidata.zip;
http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip  
  
Q2.1: What does it mean "The number of the training set is 𝑚 − 3, with the first 𝑚′ ∈ [1,𝑚 − 3] check-ins as input sequence and the [2,𝑚 − 2]-nd visited location as the label"?  
A2.1: We use [1] as input to predict [2], use [1,2] as input to predict [3], and ..., until we use [1,...,m-3] to predict [m-2]. Basically we do not use the last few steps and reserve them as a simulation of "future visits" to test the model since these last steps are not fed into the model during training.  
  
Q2.2: Can you please explain your trajectory encoding process? Do you create the location embeddings using skip-gram-like approaches?  
A2.2: Pre-training of embedding is an effective approach and can further improve the performance for sure. Unfortunately, the focus and contribution of this paper are not on embedding pre-training but on spatio-temporal linear embedding, and pretraining is not used in baselines, so we do not use it in our paper.

Q2.3: Would it be better to construct edges based on spatial distances instead of using distances?  
A2.3: If the edges can truly reflect the relations between each loaction and each user, then yes. Ideal 0-1 edge relation is a stronger representation. However, constructing edges merely based on spatial distances can raise problems. Consider that a 30-kilometer metro takes less time than a 5-kilometer walk. From the data, we only know distances.  

Q2.4: What do you mean by setting a unit spatiotemporal embedding?  
A2.4: ![image](https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation/blob/master/unit_embedding.png)

Q2.5: What does each column/row in NYC.npy mean?  
A2.5: Each row: [user id, check-in location id, time in minutes].  

Q2.6: Can we try a different division of train/dev/test datasets?  
A2.6: Our goal here is to generalize for the future visits of each user we have known (we do not want to test the model performance on biased past behavior), instead of generalizing to other users whose user-id embeddings are not known to the model. 

Q2.7: How is the value of the recall rate calculated in your paper? For example, the top5 probability of the NYC data set is 0.xx but in the paper it is 0.xxxx.  
A2.7: It is common practice to run under different seeds and get the average value. We averaged the ten times results and all of them are accepted by the statistical test of p=0.01. 

Q3: What is the environment to run the code? And version?  
A3: We use python 3.7.2, CUDA 10.1 and PyTorch 1.7.1. Make sure to install all libs that we import.  
