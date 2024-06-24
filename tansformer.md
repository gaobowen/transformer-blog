# Tansformer

## 简介
Transformer是一种深度学习模型，它在自然语言处理（NLP）领域中非常流行和有效。它最初由Vaswani等人在2017年的论文《Attention is All You Need》中提出。Transformer模型的核心思想是使用自注意力（Self-Attention）机制来处理序列数据，这使得模型能够在处理长距离依赖关系时表现出色。  

近年来，Transformer不仅仅局限于NLP领域，已经广泛应用于 图像识别(ViT)、图像生成(DiT)、语音识别(Hubert)、大语言模型(ChatGPT)、智能驾驶、人形机器人等领域。其中的注意力模块已经是各种大型深度网络中的标准模块。

![](https://pic1.zhimg.com/v2-4b53b731a961ee467928619d14a5fd44_r.jpg)





## 扩展
### MoE 混合专家模型  
![](https://pic1.zhimg.com/80/v2-1914a0e3f9f670af1fc33f59d5f41024_720w.webp)  
### ViT (EncoderOnly)  
从图片到特征
![](https://img-blog.csdnimg.cn/direct/043b874f1cf94335b09e8498351ab3b4.png#pic_center)  
### [LlamaGen](https://github.com/FoundationVision/LlamaGen/tree/main) (DecoderOnly)
从特征生成图片
![](https://peizesun.github.io/llamagen/Vanilla_files/figs/text-conditional.png)










