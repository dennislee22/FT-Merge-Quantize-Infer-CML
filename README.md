LLM: Fine-Tune > Merge > Quantize > Infer .. on CML
===

## <a name="toc_0"></a>Table of Contents
[//]: # (TOC)
[1. Objective](#toc_0)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[1.1. Benchmark Summary](#toc_2)<br>
[2. Preparation](#toc_3)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.1. Python Libraries](#toc_4)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.2. Dataset](#toc_5)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.3. Model](#toc_6)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2.4. Infrastructure](#toc_7)<br>
[3. bigscience/bloom-1b1](#toc_8)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.1. Fine-Tune & Merge](#toc_9)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.2. Quantize](#toc_10)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.3. Inference](#toc_11)<br>
[4. bigscience/bloomz-7b1](#toc_12)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.1. Fine-Tune & Merge](#toc_13)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.2. Quantize](#toc_14)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.3. Inference](#toc_15)<br>
[5. tiiuae/falcon-1b](#toc_16)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.1. Fine-Tune & Merge](#toc_17)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.2. Quantize](#toc_18)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.3. Inference](#toc_19)<br>
[6. tiiuae/falcon-7b](#toc_20)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.1. Fine-Tune & Merge](#toc_21)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.2. Quantize](#toc_22)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.3. Inference](#toc_23)<br>
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

#### <a name="toc_2"></a>1.1. Benchmark Summary

- Table shows the benchmark result of fine-tuning the specific model with **Text-to-SQL** dataset.
  
| Model | Training | Duration | 
| :---      |     :---:      |   ---: |
| bloom-1b  | No quantization     | sec   |
| bloom-1b  | BitsAndBytes      | sec     |

### <a name="toc_3"></a>2. Preparation


### <a name="toc_3"></a>3. bigscience/bloom-1b1

#### <a name="toc_3"></a>3.1. Fine-Tune & Merge
This tool will remove any numbers after the "#" heading markers and replacing them with calculated ones

## <a name="toc_4"></a>3.2. Quantize


## <a name="toc_4"></a>3.2. Inference


Objective
----


Preparation
----
1. Install the Python libraries

```shell
pip -r -U requirements.txt
```

- The quantization requires sample data to calibrate and enhance quality of the quantization. In this benchmark test, [C4 dataset]([https://nodejs.org/en/](https://huggingface.co/datasets/c4)) is utilized. It is a large-scale, multilingual collection of web text gathered from the Common Crawl project. A quick check at the Open LLM Leaderboard reveals that performance degradation is quite minimal.

<img width="1235" alt="image" src="https://github.com/dennislee22/Quantization-LLM/assets/35444414/3f8eb810-1ec4-4b78-af99-e918d6ebb9c5">


C4 (Colossal Clean Crawled Corpus) dataset to generate our samples. The C4 dataset is a large-scale, multilingual collection of web text gathered from the Common Crawl project.

#### Fine-tune 'Falcon-1B' with text-to-SQL dataset using TRL and PEFT (FP32):

- Code snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- During training:

<img width="964" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/99a2ec1a-4cd8-4838-8b9a-5e2771a2c873">

- Time taken to train:

```
{'loss': 0.4692, 'learning_rate': 0.00019100704165606177, 'epoch': 2.04}
{'loss': 0.5298, 'learning_rate': 0.00019096462204123187, 'epoch': 2.04}
{'loss': 0.4378, 'learning_rate': 0.00019092220242640197, 'epoch': 2.05}
{'train_runtime': 1219.2765, 'train_samples_per_second': 19.334, 'train_steps_per_second': 19.334, 'train_loss': 0.5385711596014342, 'epoch': 2.05}
Training Done
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

- Parameters info of the merged model:
<img width="1022" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/4eaea460-1035-401c-9395-c1ee0f14657d">
<img width="965" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/e6b1a502-fbc1-465a-a754-638da8f9ab29">

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
