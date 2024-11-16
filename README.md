# GlyphicVLM-for-ChineseEE
Code and data for EMNLP2024 Findings Paper: 

Employing Glyphic Information for Chinese Event Extraction with Vision-Language Model.

Xiaoyi Bao, Jinghang Gu, Zhongqing Wang, Minjie Qiang, and Chu-Ren Huang. 2024.  In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 1068–1080, Miami, Florida, USA. Association for Computational Linguistics.


## Data Preparation and Preprocessing

### ACE2005
We adopt the spliting and proprocessing rom ONEIE (https://github.com/GerlinGreen/OneIE/tree/main), please follow specific instruction in ONEIE.

### KBP2017
We adopt the KBP2017 data from  (https://arxiv.org/pdf/2012.01878).

The raw data will be processed into the formation shown in the  [example file](raw_data/raw_data_example.json).


## Data Preparation and Preprocessing

### ACE2005
We adopt the spliting and proprocessing rom ONEIE (https://github.com/GerlinGreen/OneIE/tree/main), please follow specific instruction in ONEIE.

### KBP2017
We adopt the KBP2017 data from  (https://arxiv.org/pdf/2012.01878).

The raw data will be processed into the formation shown in the  [example file](raw_data/raw_data_example.json).

If you do not have access to the two above dataset, please  [email](p2213545413@outlook.com) me after obtaining the licence.


## Pre-requirement

    git clone  https://github.com/HoraceXIaoyiBao/GlyphicVLM-for-ChineseEE.git
    conda create -n gvlm python=3.9
    conda activate gvlm

    pip3 install torch torchvision torchaudio
    pip install transformers==4.30.2 timm==0.4.12 sentencepiece==0.1.99 gradio==3.44.4 markdown2==2.4.10 xlsxwriter==3.1.2 einops deepspeed peft
    


## Glyphic Image Generation

For printing the visual emphasises, we need the gold label for train set and the LLM-predicted label for the test set.

We provide the LLM-predicted label in [ACE2005_test_predict](raw_data/generated_predictions.jsonl), which is annotated by a LoRA finetuned LLaMA-3-8B and align with our processed ACE05 testset. 

You can also finetune and predict it by yourself.


Place the train/dev/test json files in the raw_data folder, named zh_train_seq_oneie.json zh_dev_seq_oneie.json zh_test_seq_oneie.json

    cd data
    python getimage_tradition_color.py
    python getimage_tradition_color_dev.py
    python getimage_tradition_color_test.py

    python ./final_oneie_tradition/turn_glm_intern.py


## Train
Modify the data_path and output_dir in  [finetune_lora.sh](/finetune/finetune_lora.sh) and run
 
    sh finetune_lora.sh


## Test & Infer
Modify the output_checkpoint in  [infer_search.py](/infer_search.py) and run
 
    python infer_search.py

