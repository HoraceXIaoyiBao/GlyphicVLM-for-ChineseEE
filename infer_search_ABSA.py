import torch
from transformers import AutoModel, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import json
from tqdm import tqdm
import time

# print("sleep for 4 hours")
# time.sleep(14400)
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-start', type=int, default=0,
                    help='start infer checkpoint')
parser.add_argument('-end', type=int, default=-1,
                    help='end infer checkpoint')


args = parser.parse_args()

#获得传入的参数
print(args)

import os

print()

checkpoints=os.listdir("/mnt/4dcc8983-1f7f-437e-bbca-b132b06be738/baoxiaoyi/ARRAUG2024/twitter2017_predict")

checkpoints=["/mnt/4dcc8983-1f7f-437e-bbca-b132b06be738/baoxiaoyi/ARRAUG2024/twitter2017_predict/"+ i for i in checkpoints if 'checkpoint' in i]
checkpoints=sorted(checkpoints, key= lambda x:int(x.split("-")[-1]))

print(checkpoints)

tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)

if args.end==-1:
  checkpoints=checkpoints[args.start:]
else:
  checkpoints=checkpoints[args.start:args.end]
  
for c in checkpoints:
  print(c)
for c in checkpoints:
  torch.set_grad_enabled(False)

  print(c)
  
  model = AutoPeftModelForCausalLM.from_pretrained(
      # path to the output directory
      c,
      trust_remote_code=True
  ).half().cuda().eval()

  # model = AutoModel.from_pretrained('/home/xybao/emnlp2024/InternLM-XComposer-main/checkpoint/checkpoint-23625',trust_remote_code=True).half().cuda().eval()
 


  # tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)



  f=open("/home/baoxiaoyi/ACL_Arr_aug/final_data/twitter2017_test_num_internLM.json","r")

  data=json.load(f)
  f.close()


  f=open("/home/baoxiaoyi/emnlp2024/InternLM-XComposer/infer_ABSA/out_twitter2017_predict_"+ c.split("-")[-1] +".json","w")


  for i in tqdm(data):
    
    text = i["conversations"][0]["value"]
    #'<ImageHere> 埃斯特拉达在接受三家电视台的访问时表示，他这位过 去 的酒友，也就是省长辛森曾经将从非法赌博业者所取得的汇款 存入一间银行的账户。'
    
    image = i["image"][0]
    
    label = i["conversations"][1]["value"]
    #'/home/baoxiaoyi/emnlp2024/InternLM-XComposer-main/final/zh-test-seq_ee_15.jpg'
    # with torch.cuda.amp.autocast():
    with torch.cuda.amp.autocast():
      predict, _ = model.chat(tokenizer, query=text, image=image, history=[], do_sample=False)
    
    
    final=dict()
    final["prompt"]=text
    final["image"]=image
    final["label"]=label
    final["predict"]=predict
    
    json_str=json.dumps(final,ensure_ascii=False)
    
    f.write(json_str+"\n")
  f.close()
  
  
  del model
  torch.cuda.empty_cache()
  
  