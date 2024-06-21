# 3.4 softmax回归
# 理论介绍，没有代码
# 精彩节选
"""信息量
压缩与预测有什么关系呢?想象一下，我们有一个要压缩的数据流。如果我们很容易预测下一个数据，
那么 这个数据就很容易压缩。
为什么呢?举一个极端的例子，假如数据流中的每个数据完全相同，这会是一个非常无聊的数据流。
由于它们总是相同的，我们总是知道下一个数据是什么。
所以，为了传递数据流的内容，我们不必传输任何信息。
也就是说，“下一个数据是xx”这个事件毫无信息量。
但是，如果我们不能完全预测每一个事件，那么我们有时可能会感到“惊异”。
克劳德·香农决定用信息量log(1/P(j))=−logP(j)来量化这种惊异程度。
在观察一个事件j时，并赋予它(主观)概率P(j)。
当我们赋P(j)予一个事件较低的概率时，我们的惊异会更大，该事件的信息量也就更大。
在 (3.4.11)中定义的熵，是当分配的概率真正匹配数据生成过程时的信息量的期望。
"""

"""
重新审视交叉熵
如果把熵H(P)想象为“知道真实概率的人所经历的惊异程度”，那么什么是交叉熵?
交叉熵从P到Q，记 为H(P, Q)。
我们可以把交叉熵想象为“主观概率为Q的观察者在看到根据概率P生成的数据时的预期惊异”。 
当P = Q时，交叉熵达到最低。在这种情况下，从P到Q的交叉熵是H(P,P) = H(P)。
简而言之，我们可以从两方面来考虑交叉熵分类目标:
(i)最大化观测数据的似然;(ii)最小化传达标签所需的惊异。
"""
