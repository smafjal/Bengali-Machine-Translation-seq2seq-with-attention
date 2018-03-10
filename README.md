# Bengali Machine Translation (ENG-BEN)

This repo contains all my works to solve the task English to Bengali translation.

One of the greatest idea was Sequence to sequence netwok (seq2seq) where two Recurrent Neural Network (RNN) used to transform one sequence to another sequence. An RNN-Encoder embeded sequence of words in a vector space where another RNN-Decoder try to decode the encoded sequence. 

Another improvment added called attention mechanism which lets to decoder focus over a specific range of the input sequence.

#### Train
To train the model put your data at ``data`` folder. Sequence will be space separated ``INPUT-LAN <space> OUTPUT-LAN``

Run ``python3.6 train.py`` for training.

#### Test
To evalute model run ``python3.6 eval.py``

#### Dependencies
1. Pytorch 0.3.0
2. Numpy

#### Thanks
1. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
2. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)



