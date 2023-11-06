LLM: Fine-Tune > Merge > Quantize > Infer .. on CML
===

## <a name="toc_0"></a>Table of Contents
[//]: # (TOC)
[1. Objective](#toc_0)<br>
[2. Summary & Benchmark Score](#toc_2)<br>
[3. Preparation](#toc_3)<br>
[4. bigscience/bloom-1b1](#toc_4)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.1. Fine-Tune > Merge > Inference](#toc_5)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.2. Quantize > Inference](#toc_6)<br>
[5. bigscience/bloomz-7b1](#toc_7)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.1. Fine-Tune > Merge > Inference](#toc_8)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.2. Quantize > Inference](#toc_9)<br>
[6. tiiuae/falcon-1b](#toc_10)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.1. Fine-Tune > Merge > Inference](#toc_11)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.2. Quantize > Inference](#toc_12)<br>
[6. tiiuae/falcon-7b](#toc_20)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.1. Fine-Tune > Merge > Inference](#toc_11)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.2. Quantize > Inference](#toc_12)<br>
[7. Salesforce/codegen2-1B](#toc_24)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7.1. Fine-Tune & Merge](#toc_25)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7.2. Quantize](#toc_26)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7.3. Inference](#toc_27)<br>

[//]: # (/TOC)

### <a name="toc_0"></a>1. Objective

- In the event that you have limited GPU resources or even have no GPU in your infrastructure landscape, you may run your GenAI application using quantized models. This articles focuses on how to quantize your language models in 8, 4, or even 2 bits without **significant** performance degradation and quicker inference speed, with the help of Transformers API.
GPTQ, a Post-Training Quantization (PTQ) technique.
- GPTQ adopts a mixed int4/fp16 quantization scheme where weights are quantized as int4 while activations remain in float16. During inference, weights are dequantized on the fly and the actual compute is performed in float16.
- bitsandbytes (zero-shot quantization)

- To comprehend this, it’s crucial to realize that during model training, the model states are the main contributors to memory usage. These include tensors composed of optimizer states, gradients, and parameters. In addition to these model states, there are activations, temporary buffers, and fragmented memory, collectively known as residual states, that consume the remaining memory.

- The latest way to train big models using the newest NVIDIA graphics cards uses a method known as mixed-precision (FP16/32) training. FP32 is called full precision (4 bytes), while FP16 are referred to as half-precision (2 bytes). Here, important model components like parameters and activations are stored as FP16. This storage method allows these graphics cards to process large amounts of data very quickly.

During this training process, both the forward and backward steps are done using FP16 weights and activations. However, to properly calculate and apply the updates at the end of the backward step, the mixed-precision optimizer keeps an FP32 copy of the parameters and all other states used in the optimizer.



#### <a name="toc_2"></a>2. Summary & Benchmark Score

- Table shows the benchmark result of fine-tuning the specific model with **Text-to-SQL** dataset.
  
| Model | Training | Duration | 
| :---      |     :---:      |   ---: |
| bloom-1b  | No quantization     | sec   |
| bloom-1b  | BitsAndBytes      | sec     |

### <a name="toc_3"></a>2. Preparation

1. Install the Python libraries

```shell
pip -r -U requirements.txt
```

- The quantization requires sample data to calibrate and enhance quality of the quantization. In this benchmark test, [C4 dataset](https://huggingface.co/datasets/c4) is utilized. It is a large-scale, multilingual collection of web text gathered from the Common Crawl project. A quick check at the Open LLM Leaderboard reveals that performance degradation is quite minimal.


### <a name="toc_3"></a>3. bigscience/bloom-1b1

#### <a name="toc_3"></a>3.1. Fine-Tune (wo Quantization) > Merge > Inference

- Code Snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- Load model before fine-tuning/training starts:
```
Base Model Memory Footprint in VRAM: 4063.8516 MB
--------------------------------------
Parameters loaded for model bloom-1b1:
Total parameters: 1065.3143 M
Trainable parameters: 1065.3143 M


Data types for loaded model bloom-1b1:
torch.float32, 1065.3143 M, 100.00 %
```

<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/4e557656-abb9-409f-8a56-23601af785f9">

- During fine-tuning/training:
<img width="974" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/d1a594f6-c284-4cb7-bda5-17d19227626d">

```
{'loss': 0.8376, 'learning_rate': 0.0001936370577755154, 'epoch': 2.03}
{'loss': 0.7142, 'learning_rate': 0.0001935522185458556, 'epoch': 2.03}
{'loss': 0.6476, 'learning_rate': 0.00019346737931619584, 'epoch': 2.03}
{'train_runtime': 715.2236, 'train_samples_per_second': 32.96, 'train_steps_per_second': 16.48, 'train_loss': 0.8183029612163445, 'epoch': 2.03}
Training Done
```
- After training is completed, merge the base model with the PEFT-trained adapters.
- Load the merged model:
```
Merged Model Memory Footprint in VRAM: 4063.8516 MB

Data types:
torch.float32, 1065.3143 M, 100.00 %
```
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/021a1854-f943-4257-9165-f90bde98c5e8">

- Run inference on the fine-tuned/merged model and the base model:
```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:

--------------------------------------
Fine-tuned Model Result :
SELECT Title FROM book WHERE Writer <> 'Dennis Lee'
```

```
Base Model Result :
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is Dennis Lee?
# result:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is Dennis Lee?
```

#### <a name="toc_4"></a>3.2. Quantize > Inference
- During quantization:
<img width="1059" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/414dca58-025a-48b2-93e4-816b5781e0ce">

<img width="974" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/218470a5-4358-41ce-8661-0dc8b21bf224">

- Time taken to quantize:
```
Total Seconds Taken to Quantize Using cuda:0: 282.6761214733124
```

- Loaded quantized model:
```
cuda:0 Memory Footprint: 1400.0977 MB

Data types:
torch.float16, 385.5053 M, 100.00 %

```
<img width="975" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/75965ac1-81ce-4c5e-8aca-83246cf674ab">

- Infer quantized model:
```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:

--------------------------------------
Quantized Model Result :
SELECT Title FROM book WHERE Writer = 'Not Dennis Lee'
```

### <a name="toc_20"></a>6. tiiuae/falcon-7b

#### <a name="toc_3"></a>3.1. Fine-Tune (wo Quantization) > Merge > Inference

- Code Snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- Load model before fine-tuning/training starts:
```
Base Model Memory Footprint in VRAM: 26404.2729 MB
--------------------------------------
Parameters loaded for model falcon-7b:
Total parameters: 6921.7207 M
Trainable parameters: 6921.7207 M


Data types for loaded model falcon-7b:
torch.float32, 6921.7207 M, 100.00 %
```
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/dd499a83-f7b9-41d3-8c59-955e9e16a0fc">

- During fine-tuning/training:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 1.11 GiB. GPU 0 has a total capacty of 39.39 GiB of which 345.94 MiB is free. Process 1618370 has 39.04 GiB memory in use. Of the allocated memory 37.50 GiB is allocated by PyTorch, and 1.05 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/0e91da7b-f704-4b03-a824-b5391819a6c8">

#### <a name="toc_3"></a>3.1. Fine-Tune (w 8-bit Quantization) > Merge > Inference

- Code Snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- Load model before fine-tuning/training starts:
```
Base Model Memory Footprint in VRAM: 6883.1384 MB
--------------------------------------
Parameters loaded for model falcon-7b:
Total parameters: 6921.7207 M
Trainable parameters: 295.7690 M


Data types for loaded model falcon-7b:
torch.float16, 295.7690 M, 4.27 %
torch.int8, 6625.9517 M, 95.73 %
```
<img width="974" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/a228e9ef-6e4d-438b-8b3a-53d74f7d127b">

- During fine-tuning/training:
<img width="975" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/6a743e35-672f-4163-916b-0b491d88bf42">

```
warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
```

```

```
- After training is completed, merge the base model with the PEFT-trained adapters.
- Load the merged model:
```
Merged Model Memory Footprint in VRAM: 4063.8516 MB

Data types:
torch.float32, 1065.3143 M, 100.00 %
```
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/021a1854-f943-4257-9165-f90bde98c5e8">

- Run inference on the fine-tuned/merged model and the base model:
```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:

--------------------------------------
Fine-tuned Model Result :
SELECT Title FROM book WHERE Writer <> 'Dennis Lee'
```

```
Base Model Result :
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is Dennis Lee?
# result:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is Dennis Lee?
```

#### <a name="toc_4"></a>3.2. Quantize > Inference
- During quantization:
<img width="1059" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/414dca58-025a-48b2-93e4-816b5781e0ce">

<img width="974" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/218470a5-4358-41ce-8661-0dc8b21bf224">

- Time taken to quantize:
```
Total Seconds Taken to Quantize Using cuda:0: 282.6761214733124
```

- Loaded quantized model:
```
cuda:0 Memory Footprint: 1400.0977 MB

Data types:
torch.float16, 385.5053 M, 100.00 %

```
<img width="975" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/75965ac1-81ce-4c5e-8aca-83246cf674ab">

- Infer quantized model:
```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:

--------------------------------------
Quantized Model Result :
SELECT Title FROM book WHERE Writer = 'Not Dennis Lee'
```

#### <a name="toc_21"></a>6.1. Fine-Tune & Merge

```
cuda:0 Memory Used: 3723.6384 MB
--------------------------------------

Parameters loaded for model falcon-7b:
Total parameters: 3608.7448 M
Trainable parameters: 295.7690 M


Data types for loaded model falcon-7b:
torch.float16, 295.7690 M, 8.20 %
torch.uint8, 3312.9759 M, 91.80 %
```

```
{'loss': 0.5093, 'learning_rate': 0.0001921099516416391, 'epoch': 2.04}
{'loss': 0.4912, 'learning_rate': 0.00019202511241197932, 'epoch': 2.04}
{'loss': 0.524, 'learning_rate': 0.00019194027318231952, 'epoch': 2.04}
{'train_runtime': 6659.1722, 'train_samples_per_second': 3.54, 'train_steps_per_second': 1.77, 'train_loss': 0.5437259482040565, 'epoch': 2.04}
Training Done
```

#### <a name="toc_"></a>6.2. Quantize


```
quantization_config = GPTQConfig(bits=8, dataset = "c4", tokenizer=tokenizer, disable_exllama=True)
```

```
Total Seconds Taken to Quantize Using cuda:0: 1384.6443202495575
```

```
$ ls -lh gptq-merged_falcon-7b_8bit
total 6.9G
-rw-r--r--. 1 cdsw cdsw 1.7K Nov  1 06:54 config.json
-rw-r--r--. 1 cdsw cdsw  118 Nov  1 06:54 generation_config.json
-rw-r--r--. 1 cdsw cdsw 4.7G Nov  1 06:54 pytorch_model-00001-of-00002.bin
-rw-r--r--. 1 cdsw cdsw 2.3G Nov  1 06:54 pytorch_model-00002-of-00002.bin
-rw-r--r--. 1 cdsw cdsw  61K Nov  1 06:54 pytorch_model.bin.index.json
-rw-r--r--. 1 cdsw cdsw  541 Nov  1 06:54 special_tokens_map.json
-rw-r--r--. 1 cdsw cdsw 2.6K Nov  1 06:54 tokenizer_config.json
-rw-r--r--. 1 cdsw cdsw 2.7M Nov  1 06:54 tokenizer.json
```

```
$ ls -lh gptq-merged_falcon-7b_4bit
total 3.8G
-rw-r--r--. 1 cdsw cdsw 1.7K Nov  1 05:42 config.json
-rw-r--r--. 1 cdsw cdsw  118 Nov  1 05:42 generation_config.json
-rw-r--r--. 1 cdsw cdsw 3.8G Nov  1 05:42 pytorch_model.bin
-rw-r--r--. 1 cdsw cdsw  541 Nov  1 05:42 special_tokens_map.json
-rw-r--r--. 1 cdsw cdsw 2.6K Nov  1 05:42 tokenizer_config.json
-rw-r--r--. 1 cdsw cdsw 2.7M Nov  1 05:42 tokenizer.json
```



#### <a name="toc_23"></a>6.2. Inference

8-bit Parameter Precision Info:
```
cuda:0 Memory Footprint: 7038.3259 MB
Total parameters: 295.7690 M
Trainable parameters: 295.7690 M

Data types:
torch.float16, 295.7690 M, 100.00 %
```

8-bit gpustat:
```
[0] NVIDIA A100-PCIE-40GB | 29°C,   0 % |  8097 / 40960 MB |
```
8-bit config.json:
```
  "quantization_config": {
    "batch_size": 1,
    "bits": 8,
    "block_name_to_quantize": "transformer.h",
    "damp_percent": 0.1,
    "dataset": "c4",
    "desc_act": false,
    "disable_exllama": true,
    "group_size": 128,
    "max_input_length": null,
    "model_seqlen": 2048,
    "module_name_preceding_first_block": [
      "transformer.word_embeddings"
    ],
    "pad_token_id": null,
    "quant_method": "gptq",
    "sym": true,
    "tokenizer": null,
    "true_sequential": true,
    "use_cuda_fp16": true
  },
```

#### Fine-tune 'Falcon-1B' with text-to-SQL dataset using TRL and PEFT (FP32):

- Code snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- During training:


- Time taken to train:

```

```

- Merged files:
  
```
$ ls -lh merged_falcon-rw-1b
**total 4.9G**
-rw-r--r--. 1 cdsw cdsw 1.2K Nov  1 01:58 config.json
-rw-r--r--. 1 cdsw cdsw  116 Nov  1 01:58 generation_config.json
-rw-r--r--. 1 cdsw cdsw 446K Nov  1 01:58 merges.txt
-rw-r--r--. 1 cdsw cdsw 4.7G Nov  1 01:58 pytorch_model-00001-of-00002.bin
-rw-r--r--. 1 cdsw cdsw 257M Nov  1 01:58 pytorch_model-00002-of-00002.bin
-rw-r--r--. 1 cdsw cdsw  25K Nov  1 01:58 pytorch_model.bin.index.json
-rw-r--r--. 1 cdsw cdsw  131 Nov  1 01:58 special_tokens_map.json
-rw-r--r--. 1 cdsw cdsw  477 Nov  1 01:58 tokenizer_config.json
-rw-r--r--. 1 cdsw cdsw 2.1M Nov  1 01:58 tokenizer.json
-rw-r--r--. 1 cdsw cdsw 780K Nov  1 01:58 vocab.json
```


#### Quantize

- During quantization process:

<img width="1056" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/a7935a5b-3b3d-419b-8257-8635f829e4e9">

<img width="1023" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/e07355d0-6f08-4fe0-a708-40380c1323cd">

- When exllama is enabled, 'Assertion error:`
- Disabling exllama allowing the quantization process to complete. Notice that CPU is also being used:

<img width="1022" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/127c3e28-b194-407d-acef-f7f3a75b70ce">

<img width="900" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/daf4c454-4689-4e5f-99e4-5dfb5216ebfa">

```
Total Seconds Taken to Quantize Using cuda:0: 1350.0081555843353
```

```
ls -lh gptq-merged_falcon-7b_4bit
total 3.8G
-rw-r--r--. 1 cdsw cdsw 1.7K Nov  1 05:42 config.json
-rw-r--r--. 1 cdsw cdsw  118 Nov  1 05:42 generation_config.json
-rw-r--r--. 1 cdsw cdsw 3.8G Nov  1 05:42 pytorch_model.bin
-rw-r--r--. 1 cdsw cdsw  541 Nov  1 05:42 special_tokens_map.json
-rw-r--r--. 1 cdsw cdsw 2.6K Nov  1 05:42 tokenizer_config.json
-rw-r--r--. 1 cdsw cdsw 2.7M Nov  1 05:42 tokenizer.json
```

#### Model Inference

<img width="1021" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/67a1bcd2-5e62-4207-84e1-1859acb62059">

```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:
--------------------------------------
Fine-tuned Model Result :

SELECT Title FROM book WHERE Writer <> "Dennis Lee"
```
