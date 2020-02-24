文章主要提出了使用CNN可以输入的CTR实例中提取全局-局部关键信息，从而实现了单个广告曝光（single ad impression）而且也包括时序性（sequential ad impression）广告曝光。利用数据集[Avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data)和[Yoochoose](http://recsys.yoochoose.net)进行了算法的验证。

**1. 为什么FM和MF不能与之抗衡**

MF(matrix factorization)和FM(Factorization Machines)可以捕获成对特征之间的相关性关系，不能去获取更高维度的特征交互信息（即特征）。

[FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)可以解决大规模稀疏矩阵预测问题，并且不想SVM那样，FM是可以在线性时间内被计算出来，并且模型参数估计不依赖于任何其他的支持向量。与SVD++，PITF和PIMC相比，FM对于数据的容忍性较好，能够接受Null和None的元素即不需要矩阵填充。其中，FM模型定义如下：

$$\bar{y}(x):=w_0+\sum_{i=1}^nw_ix_i+\sum_{i=1}^n\sum_{j=i+1}^n<v_i,v_j>x_ix_j$$

其中$w_0 \in R$，$w \in R^n$，$V \in R^{n \times k}$，$<\cdot,\cdot>$表示两个大小为$k$矩阵的内积，即

$$<v_i,v_j>=\sum_{f=1}^kv_{i,k} \cdot v_{j,f}$$

矩阵$V$的每个样本，用大小为$k$的向量$v_i$描述其第$i$个变量，其中$k \in N^{+}$是一个超参数，表示因子机的维度。$w_0$是全局的一个偏置，$w_i$是表示第$i$个变量的权重，$\bar{w}_{i,j}=<v_i, v_j>$表示第$i$和$j$变量内积的权重，取代了原来模型学习的$w_{i,j}\in R$，这个点可以在稀疏情况下对高阶交互进行高质量的参数估计。

![](./imgs/FM-1.png)

其中每个Feature vector $ x$ 的与 Target $y$一一对应。第1个4列表示活动的用户；下一个5表示活动的项目，下一个5列可以表示用户已经评分的电影，Time表示以月为单位的时间，最后5列表示在给电影打分之前最后一次评分电影。

对于如上的稀疏矩阵如User A对Movie ST的打分，其中可以看出这两

计算复杂度计算

$$\sum_{i=1}^n\sum_{j=i+1}^n<v_i, v_j>x_ix_j $$

$$=\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n<v_i,v_j>x_ix_j-\frac{1}{2}\sum_{i=1}^n<v_i, v_i>x_ix_i$$

$$=\frac{1}{2}(\sum_{i=1}^n\sum_{j=1}^n\sum_{f=1}^k v_{i,f}v_{j,f}x_ix_j-\sum_{i=1}^n\sum_{f=1}^kv_{i,f}v_{i,f}x_ix_i)$$

$$=\frac{1}{2}\sum_{f=1}^k((\sum_{i=1}^nv_{i,f}x_i)(\sum_{j=1}^nv_{j,k}x_j)-\sum_{i=1}^nv_{j,f}^2x_i^2)$$

$$=\frac{1}{2}\sum_{f=1}^k((\sum_{i=1}^nv_{i,f}x_i)^2-\sum_{i=1}^nv_{i,f}^2x_i^2)$$

上边仅需要计算k和n次。

FM可以做逻辑回归（损失函数可以使用MSE）、二分类（使用hige loss或者logit loss）、Ranking等，参数学习可以利用SGD等方式。并且其可以进一步扩展到$d$-维的方式，

$$\bar{y}(x)=w_0+\sum_{i=1}^n w_i x_i + \sum_{l=2}^d \sum_{i_1=1}^n \cdots \sum_{i_l=i_{l-1}+1}^n (\prod_{j=1}^l x_{i_j}) (\sum_{f=1}^{k_l} \prod_{j=1}^lv_{i_j, f}^{(l)})$$



对于任何正定矩阵$W$，只要$k$足够大，就存在一个矩阵$V$，使得$W=V\cdot V^T$。这表明，若选择足够大的$k$，则FM可以表示任何交互矩阵$W$。但是，在稀疏环境中，通常应该选择一个小的$k$值。



特征值分解、SVD、MF、SVD++ 、FM、ALS 均完成了矩阵分解操作，区别在于SVD与 SVD++均需要矩阵填充， 而 funk-svd 与FM、 ALS 均采用MF分解模式，进行隐语义分解，其中ALS主要用于协同过滤，而FM主要用于ctr cvr预估，FM又解决了了特征关系问题—通过向量内积。