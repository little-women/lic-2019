# lic-2019

**Baseline 的几点思考？**

1. Baseline 使用后验之后，为什么验证和测试差那么多？
2. Baseline 选取出 k 后，decoder 的时候 怎么利用 k  的？
3. Baseline 选取出的 k 是加权后的，能否直接取得最大的 k ，然后 decoder 的时候 从 k 中直接 copy 词（怎样控制不重复的 copy 相同的词）？
4. decoder 的每一步 选取的 k 一样吗？
5. Baseline input 为 goal+src+knowledge，单独出来呢？
6. 考虑到多轮对话，encoder 的时候可以用 hiera 结构试试。



