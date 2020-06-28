# word2vec是用于产生词向量的工具
  本文解读word2vec的两篇论文，均由Tomas Mikolov提出，第一篇Efficient Estimation of Word Representations in Vector Space主要阐述word2vec的两个模型CBOW和Skip-gram，Distributed Representations of Words and Phrasesand their Compositionality则着重于skip-gram的训练和优化方法
  
  
  这里要说一下一个误区，word2vec并不是一个具体的算法或者模型，而是一个工具箱，word2vec包含的CBOW和skip-gram才是具体的模型
  
  
  由于两篇论文是同一个人写的，而且看起来像是姐妹篇，所以我就放在一起说了
  
  
- ### Efficient Estimation of Word Representations in Vector Space
  首先说第一篇，论文先介绍了一下word2vec之前提出的一些词向量产生模型，比如nnlm，rnnlm之类的
  
  
  ![nnlm2](https://github.com/jyushicelestialbeing/interpretation-of-papers/blob/master/word2vec%E4%B8%A4%E7%AF%87/nnlm2.jpg)
  
  
  nnlm模型利用一个窗口，窗口大小为n，比如我们要生成句子中第五个词的向量，可以设n=8，这样我们就取第1到4和第6到9这把个词作为输入
  
  
  论文里提到nnlm主要是为了说明不论是nnlm还是rnnlm等模型都存在着产生结果准确度不高并且算法过于复杂，开销大，效率低的问题，而主要问题则在于传统的基于全连接神经网络的语言模型不论怎样都会金国一个或多个hidden layer，由于大多数情况下单词->向量的input我们会使用one-hot编码(如果你不知道这个的话可以google一下，非常简单的一种词向量编码)，one-hot编码极为稀疏，之后将one-hot编码投射到一个投影矩阵中，在传播到hidden layer中，这会导致训练到hidden layer时hidden layer会非常巨大，下图说明了nnlm的算法复杂度
  
  
  ![nnlm](https://github.com/jyushicelestialbeing/interpretation-of-papers/blob/master/word2vec%E4%B8%A4%E7%AF%87/nnlm.jpg)
  
  
  其中N为窗口大小，D为投影矩阵维度，H为隐藏层维度，V为单词表大小(即one-hot向量维度)
  - ### CBOW
    Continuous Bag-of-Words Model 连续词袋模型
    
    
    cbow在逻辑上和nnlm很像，都是取一个窗口，利用窗口两边的n个词来计算窗口中心的词wt，区别在于cbow取消了nnlm中的hidden layer，将n个[v,1]维度的input(n个input维度则变为[v,n])直接和一个[n,d]维度共享的投影矩阵w相乘后相加得到projection层[1,d]，即为图中的projection层，之后乘一个输出矩阵[d,v]，得到output向量[1,v]，即为中心词wt的向量
    
    
    这里要注意，原论文中的复杂度其实是这样的
    ![cbow-1](https://github.com/jyushicelestialbeing/interpretation-of-papers/blob/master/word2vec%E4%B8%A4%E7%AF%87/cbow-1.jpg)
    
    
    注意到最后从projection到output的复杂度其实是[d,log2|v|]，这里其实是使用了优化方法，这个方法正好在Distributed Representations of Words and Phrasesand their Compositionality中
    
    
    CBOW架构图
    ![cbow-2](https://github.com/jyushicelestialbeing/interpretation-of-papers/blob/master/word2vec%E4%B8%A4%E7%AF%87/cbow-2.jpg)
    
    
    另外关于窗口大小的选择，原文中给出的一个比较合适的值是10
    
    
  - ### Skip-gram
    skip-gram和cbow模型正好相反，skip-gram是利用一个中心词wt去预测w1到wc+1这个c个词，总体上skip-gram的架构相同，在复杂度上，由于skip-gram的输入是一个词，所以每个单词的复杂度为D+Dxlog2|v|，但是由于最终的输出需要和每个周围词进行损失计算，所以最终实际上要训练c次，则总体复杂度为
    
    
    ![skip-gram1](https://github.com/jyushicelestialbeing/interpretation-of-papers/blob/master/word2vec%E4%B8%A4%E7%AF%87/skip-gram1.jpg)
    
    
    ![skip-gram](https://github.com/jyushicelestialbeing/interpretation-of-papers/blob/master/word2vec%E4%B8%A4%E7%AF%87/skip-gram2.jpg)
    
    
    以上是skip-gram的架构，同样在复杂度上最后的log2|v|也是使用了后面的优化方法，对于C的选择，原文中给出的比较适合的值是10
    
    
  最后要说的一点是，word2vec的训练并不要求loss低，而是只需要训练的副产物，即VxD和DxV这两个矩阵即可，由于input为one-hot编码，所以实际上这两个矩阵的每一行就代表单词表中的一个单词
  
  
  比如input为[0,1,0,0]，训练后的VxD矩阵中的第二行就代表input的这个单词，此时我们取第二行即是训练好的词向量，至于使用两个矩阵中的哪个，其实都可以，大多数情况下我们使用第一个矩阵
    
  
  最后word2vec通过一个softmax输出
- ### Distributed Representations of Words and Phrasesand their Compositionality
  本篇论文主要阐述在skip-gram架构下如何做优化
  
  
  word2vec在最后softmax输出的部分会遇到一个复杂度爆炸的问题
  
  
  ![softmax](https://github.com/jyushicelestialbeing/interpretation-of-papers/blob/master/word2vec%E4%B8%A4%E7%AF%87/softmax.jpg)
  
  
  最终softmax层计算量和词典大小V相同，一般情况下词典大小都会很大，所以当v很大时最后一层的计算几乎是不可能完成的
  
  
  Mikolov在本文中提出了两种针对word2vec的优化方法使得最后的softmax层计算量大大减少
  
  
  - ### Hierarchical Softmax
    我们将输出层变为一颗霍夫曼树，该树共有V个叶子节点，代表词典中全部的单词，共有V-1个非叶子节点
    
    
    
    
