# lic-2019

**Baseline 的几点思考？**

1. Baseline 使用后验之后，为什么验证和测试差那么多？
2. Baseline 选取出 k 后，decoder 的时候 怎么利用 k  的？
3. Baseline 选取出的 k 是加权后的，能否直接取得最大的 k ，然后 decoder 的时候 从 k 中直接 copy 词（怎样控制不重复的 copy 相同的词）？
4. decoder 的每一步 选取的 k 一样吗？
5. Baseline input 为 goal+src+knowledge，单独出来呢？
6. 考虑到多轮对话，encoder 的时候可以用 hiera 结构试试。

**数据的几点思考？**

1. robot 是主导说话的一方，根据对话历史（上一句话）预测下一句话不适用，而更多的是根据 goal path 和 knowledge 预测下一句话。

**Knowledge 表示？**

1. 每个 k 当成 seq，用 rnn encoder 成一个 vector。
2. 每个 k 的每个词单独表示，k 表示为每个词 embedding 相加。
3. 三元组表示（问题是 长句子 怎么处理？)



**Maybe?**

1. 引入 隐变量（latent variable) 表示 聊天 goal 和 path 状态

