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
    
    
    
    
    
    
