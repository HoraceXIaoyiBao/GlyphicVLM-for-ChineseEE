import json
import os


filelist=["ee_nl_zh_train_seq_oneie.json"
          "ee_nl_zh_dev_seq_oneie.json"
          "ee_nl_zh_test_seq_oneie.json"]

for filename in filelist:
  
  f=open("./final_oneie_tradition/"+filename,"r")

  data=json.load(f)
  f.close()


  internLM=[]

  for index, i in enumerate(data):
      single=dict()
      
      single["id"]=str(index)
      single["image"]=[ os.getcwd()+"/"+i["img"] ]
      
      single["conversations"]=[
          {
            "from": "user",
            "value": "<ImageHere> "+ i["prompt"]
          },
          {
            "from": "assistant",
            "value": i["label"]
          }
        ]
      
      internLM.append(single)
      
  f=open(os.getcwd()+"/final_oneie_tradition/"+filename.split(".")[0] +"_internLM.json","w")
      
  json.dump(internLM,f,indent=4,ensure_ascii=False)
  f.close()
  
f= open("./final_oneie_tradition/ace05.txt")
f.write( os.getcwd()+"/final_oneie_tradition/ee_nl_zh_train_seq_oneie_internLM.json 7\n")
f.close()
