<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

# 推理和部署

推理所需的模型权重、源码、配置已发布在 Hugging Face：[Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base) 和 [Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)。下面以 Baichuan-13B-Chat 为例示范多种推理方式。程序会自动从 Hugging Face 下载所需资源。

推理前请安装依赖：
```shell
pip install -r requirements.txt
```

## Python代码方式

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from transformers.generation.utils import GenerationConfig
>>> tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-13B-Chat", use_fast=False, trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
>>> model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan-13B-Chat")
>>> messages = []
>>> messages.append({"role": "user", "content": "世界上第二高的山峰是哪座"})
>>> response = model.chat(tokenizer, messages)
>>> print(response)
乔戈里峰。世界第二高峰———乔戈里峰西方登山者称其为k2峰，海拔高度是8611米，位于喀喇昆仑山脉的中巴边境上
```

> 在上述代码中，模型加载指定 `device_map='auto'`，会使用所有可用显卡。如需指定使用的设备，可以使用类似 `export CUDA_VISIBLE_DEVICES=0,1`（使用了0、1号显卡）的方式控制。


## 数据集准备

### 预训练数据集格式

输入数据为放置在项目`data`目录下的 txt文件，用`--dataset`选项指定（参考下面示例），多个输入文件用`,`分隔。
使用--dataset参数制定前，需要在 `data`目录 下的  'dataset_info.json' 文件中设置数据集名和路径信息

预训练数据集的处理方法在`data/pro_pre_data`目录下

1.split_ali 基于阿里的分段模型实现，具体操作参考：https://modelscope.cn/models/damo/nlp_bert_document-segmentation_chinese-base/summary

2.langchain_split 基于langchain 实现字符串分段，采用重叠设置，增强上下文语义关联
注意：由于框架使用固定token长度进行预训练数据划分，因此修改了数据加载部分的源码 ，路径 'src/llmtuner/dsets/preprocess.py'


### 微调数据集格式

输入数据为放置在项目`data`目录下的 json 文件，用`--dataset`选项指定（参考下面示例），多个输入文件用`,`分隔。
使用--dataset参数制定前，需要在 `data`目录 下的  'dataset_info.json' 文件中设置数据集名和路径信息
json 文件示例格式和字段说明如下：
```
[
    {
        "instruction": "What are the three primary colors?",
        "input": "",
        "output": "The three primary colors are red, blue, and yellow."
    },
    ....
]
```
json 文件中存储一个列表，列表的每个元素是一个 sample。其中`instruction`代表用户输入，`input`是可选项，如果开发者同时指定了`instruction`和`input`，会把二者用`\n`连接起来代表用户输入；`output`代表期望的模型输出。



## 命令行工具方式
### 预训练
参考 'pretrain.sh' 脚本， 使用deepspeed进行多卡加速
具体操作：
1.accelerate config # 首先配置分布式环境

```
(baichuan) [root@gpt-4 LLaMA-Efficient-Tuning-main]# accelerate config
[2023-08-24 16:52:10,287] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
----------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Do you wish to optimize your script with torch dynamo?[yes/NO]:no
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: no
----------------------------------------------------------------------------------------------------------------------------------------
What should be your DeepSpeed's ZeRO optimization stage?
3
----------------------------------------------------------------------------------------------------------------------------------------
Where to offload optimizer states?
none
----------------------------------------------------------------------------------------------------------------------------------------
Where to offload parameters?
none
How many gradient accumulation steps you're passing in your script? [1]: 1
Do you want to use gradient clipping? [yes/NO]: yes
What is the gradient clipping value? [1.0]: 1
Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]: yes
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: no
How many GPU(s) should be used for distributed training? [1]:4
```

2. 启动脚本
./pretrain.sh


## LoRA微调
启动脚本  ./loraFinetune.sh


## 测试
1.在测试时合并： test_predict.sh
2.先合并再使用合并后的模型测试：exportToTF.sh & test_fin.sh


