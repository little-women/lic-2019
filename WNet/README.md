# WNet

| Model        | Encoder | Decoder | F1           | BLEU 1       | BLEU 2       |
| :----------: | ------------ | ------------ | :----------: |:----------:|:---------------:|
| MemNet | EncoderMemNN | RNNDecoder | 34.48 |0.305|0.181|
| Mem2Seq | EncoderMemNN | DecoderMemNN |  |||
| PointerNet | RNNEncoder | PointerDecoder | 33.11<br />35.78 |0.267<br />0.312|0.165<br />0.185|
| DAWnet | | | |||
| CCM | | | |||
| TA-Seq2Seq | | | |||


|Baseline | F1           | BLEU 1       | BLEU 2       |
| :----------: | ------------ | ------------ | :----------: |
| Nothing changed| 35.13 |0.333|0.194|
| use_gs 改为 True | 34.99 | 0.314 | 0.186 |
| knowledge 等于最大 attn 的那个<br />use_posterior | 34.05<br />37.63 | 0.314<br />0.357 | 0.181<br />0.212 |
| knowledge 都等于 weighted_cue（KL 呈上升趋势。。。）<br />use_posterior | 33.97<br />33.97 | 0.294<br />0.294 | 0.177<br />0.177 |
| decoder 的时候 从 k 中直接 copy 词<br />use_posterior | 33.57<br />44.28 | 0.295<br />0.386 | 0.175<br />0.253 |
| decoder 的时候 从 k 中直接 copy 词 and use_posterior False | 35.89 | 0.315 | 0.190 |



