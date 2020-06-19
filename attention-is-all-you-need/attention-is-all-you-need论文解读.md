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
    ![model_architecture](https://github.com/jyushicelestialbeing/interpretation-of-the-paper/blob/master/attention-is-all-you-need/model_architecture.jpg)
    
    
    
