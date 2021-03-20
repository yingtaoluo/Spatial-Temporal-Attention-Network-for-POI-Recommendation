# STAN: Spatial-Temporal-Attention-Network-for-Next-Location-Recommendation
Update! The paper is accepted by the Web conference 2021. https://arxiv.org/abs/2102.04095

Author Reply: 
Thank you for your interest in our work! First I want to apologize for uploading the wrong files.   
Please use the new .py files in this repository if you downloaded the wrong files (before 3/18). (Sorry!)

I summarize some FAQs:  
Q1: Can you provide a dataset?  
A1: Our datasets are collected from the following links.  
http://snap.stanford.edu/data/loc-gowalla.html;  
https://personal.ntu.edu.sg/404.html; (This link was fine in the past. I do not know when and why it closed.)   
http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip  
  
Q2: I ran into some problems in reading the paper or implementing the codes. May I talk/discuss with you?  
A2: It would be my pleasure to answer your questions. Please do not hesitate to email me or leave comments at any time and explain the problem concisely so I can assist.  
  
Q2.1: What does it mean "The number of the training set is 𝑚 − 3, with the first 𝑚′ ∈ [1,𝑚 − 3] check-ins as input sequence and the [2,𝑚 − 2]-nd visited location as the label"?  
A2.1: We use [1] as input to predict [2], use [1,2] as input to predict [3], and ..., until we use [1,...,m-3] to predict [m-2].  
  
Q2.2: Can you please explain your trajectory encoding process? Do you create the location embeddings using skip-gram-like approaches?  
A2.2: Pre-training of embedding is an effective approach and can further improve the performance for sure. Unfortunately, the focus and contribution of this paper are not on embedding pre-training but on spatio-temporal linear embedding, and pretraining is not used in baselines, so we do not use it in our paper. Nevertheless, it will be a contribution if you conceive new ideas to improve embedding efficiency.  
  
Q3: What is the environment to run the code? And version?  
A3: We use python 3.7.2 with CUDA 10.1 and PyTorch 1.7.1. Make sure to install all required libs that we import.  
