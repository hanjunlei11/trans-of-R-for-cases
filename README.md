# transformer-of-Recommendation-for-similar-cases
项目简介：

给定一篇裁判文书，在数据库中检索出若干篇与之相似度最高的裁判文书。

项目方案：

由于没有相似度直接标签，所以利用远程标签来训练网络。

首先，我们人为，案件本身的信息，可以用罪名，刑期，罚金，是否财产型犯罪，是否暴力犯罪等属性来代替

所以，我们用以上属性作为该案件的标签，用属性相似来代替文本相似，具体做法是：

使用以上属性作为模型需要预测的标签，假设预测准确率极高，那我们可以人为我们得到一个拥有优秀编码能力的神经网络，该网络能够以极高的准确率将文本映射成标签。

我们将每个文本经过网络编码成的向量作为该文本的“语义”表示，保存在数据库中，这样，每篇裁判文书都得到一个代表自己语义的向量。

我们可以用余弦相似度、点积、等等可以度量相似度的方法来间接度量两个文本的相似度。

当然，如果每篇文档检索的时候都要和数据库的每个向量都计算一次相似度的话，计算量无疑是巨大的，系统响应速度将会非常缓慢。

所以我们在计算相似度之前先进行一次属性过滤，将相同属性的文书筛选出来，再进行相似度计算，计算量将会大大减少。

最后，根据相似度大小，排序将裁判文书文本输出。

各个属性的平均准确率达到94%，所以，基于我们的假设，远程标签作用在语义向量上的效果极好。
