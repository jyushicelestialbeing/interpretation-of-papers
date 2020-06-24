# Attention is All you need解读
  - ### 前言
    attention->注意力机制，目前比较火的基于deep learning技术
    
    
    本文帮助大家解读google的attention is all you need一文，项目中附上了原论文
    
    
    注意力机制用于处理序列数据(如文本等)，解决的传统rnn,lstm等模型的诸多问题
    
    
    google在原论文中称他们提出的attention模型为Transformer，完全以来self-attention机制来阐述输入和输出之间的以来关系
    
    
    Transformer也在nlp方向应用较多，并且相比传统的rnn等模型速度更快(因为它可以并行计算),训练结果质量更高
  - ### self-attention机制
    首先介绍自注意力机制，简单来说，自注意力机制就是计算序列中每个元素和其他各个元素之间的依赖关系，从而得到该元素的正确表达，这也是为什么自注意力机制相比rnn可以进行并行计算，同时和attention相比,self-attention抛弃了rnn模型(如果你对attention模型不了解的话也无妨，self-attention是attention的一种简化而非强化，所以不需要过多了解attention模型也可以看懂)
    
    
    举个例子:
    - "哎,弗洛伊德说,当儿子的都想跟自个儿妈结婚,对不?"
    
    
    如果我们以词为单位，那么"结婚"一词分别和句子中其他进行计算，最终会得出和每个词的相关性，和“弗洛伊德”的相关性可能较低，和“儿子”一词的相关性可能较高，这样得出的词向量会更加准确，word2vec中则是直接使用一个window，但某一词的前后词并不一定和当前词高度相关，这也是为什么相比rnn，self-attention不存在梯度消失的问题，另外，bert虽然也使用Transformer模型，但实际上bert在处理中文语料时使用字作为单位
    
    
  - ### Transformer架构
    ![model architecture](https://github.com/jyushicelestialbeing/interpretation-of-the-paper/blob/master/attention-is-all-you-need/model_architecture.jpg)
    
    
    左侧部分为encoder，右侧部分为decoder
    
    - encoder
    首先将词转换为词向量(word2vec等方法)作为输入放入encoder中，下文称为input，每层分为两个子层,多头self-attention和全连接网络，每个子层都使用残差连接，然后进行归一化，之后经过n层后输出，每层的输出为LayerNorm(input+SubLayer(input))
    
    在论文原文中，input的维度被设定为512(输出也为512)，n=6
    
    
    -decoder
    相比encoder，decoder多了一个mask层，共三个子层，之后会说明
    
    
    -首先说一下Transformer中的self-attention是怎么做的
    
    
    ![self attention](https://github.com/jyushicelestialbeing/interpretation-of-the-paper/blob/master/attention-is-all-you-need/self-attention.jpg)
    
    
    self-attention将input和Q,K,V三个矩阵进行计算
    
    QxK矩阵用于计算当前input和其他单词的相似度，这里要注意，实际上我们对每个单词都准备了qi与ki,单词a的qa与单词b的kb相乘的结果可以看做是单词a与b的相关度z，之后将z和vi进行计算(具体细节在下面)，即可得出当前单词的词向量
    
    
    ![QK](https://github.com/jyushicelestialbeing/interpretation-of-the-paper/blob/master/attention-is-all-you-need/QK.jpg)
    
    
    Q,K,V是怎么来的呢，实际上是通过随机初始化三个权重矩阵Wq,Wk,Wv并和input做计算训练得来的，所以Q，K，V矩阵的维度=[1,512]x[512,c] = [1,c]
    
    
    其中c是单词表大小
    
    
    最后说说计算，论文原文给出的公式如下
    
    
    ![r1](https://github.com/jyushicelestialbeing/interpretation-of-the-paper/blob/master/attention-is-all-you-need/res.jpg)
    
    
    公式本身没什么可说的了，除dk(k矩阵的维度)是直接做归一化，在这里要说一下其实公式的计算是可以直接使用矩阵加法的，也可以达到差不多的效果，但是根据实验实际上使用矩阵乘的速度要更快，所以使用了乘法，论文原文里也称这种带归一化和使用矩阵乘的self-attention为Scaled Dot-Product Attention
   
    
    -多头机制
    
    
    Transformer中使用了multi-self-attention，将Q,K,V矩阵分成多个小矩阵，论文原文中是分成64维，即分成8个，把self-attention做8次，这种方法可以捕捉不同子空间的信息，论文中说是效果更好,此时计算过程如下图所示
    
    
    ![multi-self-attention](https://github.com/jyushicelestialbeing/interpretation-of-the-paper/blob/master/attention-is-all-you-need/multi.jpg)
    
    
    最后我们将8个多头计算的结果拼接放入前向神经网络里即可
    
    
    
    
    
    
    
