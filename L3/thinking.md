## Course

### Association Rules

- Apriori
- FPGrowth
- PrefixSpan

### EDA

- Compare
- Relationship
- Composition
- Distribution

## Thinking

- 关联规则中的支持度、置信度和提升度代表的什么，如何计算？

  支持度就是几个关联的数据在数据集中出现的次数占总数据集的比重。或者说几个数据关联出现的概率。如果我们有两个想分析关联性的数据X和Y，则对应的支持度为:

  $$Support(X, Y)=P(XY)=\frac{number(XY)}{num(AllSamples)}$$

  一般来讲，支持度高的数据不一定构成频繁项集，但是支持度太低的数据肯定不构成频繁项集。

  置信度体现了一个数据出现之后，另一个数据出现的概率，或者说条件概率，若有两个分析关联性的数据$X$和$Y$，$X$对$Y$的置信度为：

  $$Confidence(X \gets Y) = P(X|Y)=\frac{P(XY)}{P(Y)}$$

  举个例子，在购物数据中，纸巾对应鸡爪的置信度为40%，支持度为1%。则意味着在购物数据中，总共有1%的用户既买鸡爪又买纸巾;同时买鸡爪的用户中有40%的用户购买纸巾。

  提升度表示含有Y的条件下，同时含有X的概率，与X总体发生的概率之比，即:

  $$Lift(X⇐Y)=P(X|Y)/P(X)=Confidence(X⇐Y)/P(X)Lift(X⇐Y)=P(X|Y)/P(X)=Confidence(X⇐Y)/P(X)$$

  提升度体先了X和Y之间的关联关系, 提升度大于1则X⇐YX⇐Y是有效的强关联规则， 提升度小于等于1则$X⇐Y$是无效的强关联规则 。一个特殊的情况，如果X和Y独立，则有$Lift(X⇐Y)=1$，因为此时$P(X|Y)=P(X)P(X|Y)=P(X)$。

- 关联规则与协同过滤的区别

  1. 关联规则面向的是**transaction**，而协同过滤面向的是**用户偏好（评分）**。
  2. 协同过滤在计算相似商品的过程中可以使用关联规则分析，但是在有用户评分的情况下（非1/0），协同过滤算法应该比传统的关联规则更能产生精准的推荐。
  3. 协同过滤的约束条件没有关联规则强，或者说更为灵活，可以考虑更多的商业实施运算和特殊的商业规则。

- 为什么我们需要多种推荐算法

  推荐算法是一种信息过滤算法，单一的算法有时候并不能保证系统的多样性以及用户的个性化

- 关联规则中的最小支持度、最小置信度如何确定

  - 试出来的

- 如果通过可视化的方式探索特征之间的相关性

  - 皮尔逊系数
  - 相关性系数
  - 热力图

## Action

- [AR-MarketBasket](https://www.kaggle.com/whs2018/marker-analysis)
- [WC-MarketBasket](https://www.kaggle.com/whs2018/marker-analysis)

## Reference

[刘建平Pinard](https://home.cnblogs.com/u/pinard/)

[知乎](https://www.zhihu.com/question/22404652)

[为什么需要推荐系统](https://blog.csdn.net/solo_ws/article/details/79455380)