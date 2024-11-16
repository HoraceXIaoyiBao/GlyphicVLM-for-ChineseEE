import pandas as pd
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import json
import unicodedata
import os

from tranS_T import *


def is_chinese(char):
    """Check if a character is a Chinese character."""
    return unicodedata.category(char).startswith('Lo')
def get_event(seq):
    events=[]
    args=[]
    
    if seq=="这个句子中不涉及任何事件。" or seq=="很抱歉，我无法理解您的问题。请提供更多上下文或详细说明。":
        return events,args
    # print(seq.replace("( Root ","").strip(")").strip())
    
    numbered_items = re.findall(r'\d+\)', seq)

    # Extract the text after each numbered item
    for item in numbered_items:
        seq=seq.replace(item,"#-#-#")
        
    event_seq=[ i.strip() for i in seq.split("#-#-#")[1:]]
    
    
    
    for i in event_seq:
        
        events_tuple=tuple(i.split("，该事件")[0].split("的事件：")) 
        events.append(events_tuple)
        
        numbered_items = re.findall(r'\[\S\]', seq)

        # Extract the text after each numbered item
        for item in numbered_items:
            i=i.replace(item,"#-#-#")
        args_seq=[ k.strip("， ") for k in i.split("#-#-#")[1:]]
        
        for j in args_seq:
            args.append( tuple(list(events_tuple)+ j.split("的") ))
            
    return tuple(args),tuple(events)


translations = {
    'Life': '生活',
    'Movement': '运动',
    'Transaction': '交易',
    'Business': '业务',
    'Conflict': '冲突',
    'Contact': '联系',
    'Personnel': '人员',
    'Justice': '正义',
    'Be-Born': '出生',
    'Marry': '结婚',
    'Divorce': '离婚',
    'Injure': '受伤',
    'Die': '死亡',
    'Transport': '运输',
    'Transfer-Ownership': '所有权转移',
    'Transfer-Money': '转账',
    'Start-Org': '成立组织',
    'Merge-Org': '合并组织',
    'Declare-Bankruptcy': '宣布破产',
    'End-Org': '终止组织',
    'Attack': '攻击',
    'Demonstrate': '示威',
    'Meet': '会面',
    'Phone-Write': '电话写作',
    'Start-Position': '起始位置',
    'End-Position': '结束位置',
    'Nominate': '提名',
    'Elect': '选举',
    'Arrest-Jail': '逮捕入狱',
    'Release-Parole': '释放假释',
    'Trial-Hearing': '审判听证',
    'Charge-Indict': '起诉',
    'Sue': '起诉',
    'Convict': '定罪',
    'Sentence': '判决',
    'Fine': '罚款',
    'Execute': '执行',
    'Extradite': '引渡',
    'Acquit': '无罪释放',
    'Appeal': '上诉',
    'Pardon': '赦免'
}

# print(translations)

def build_pool(target_seq):
    args,events=get_event(target_seq)
    # print()
    
    tri_pool=set([ i[1] for i in events])
    
    tri_pool =   "".join(list(tri_pool))
    # print(tri_pool)
    # tri_pool=set([ j for j in i[1] for i in events])
    
    # print(args)
    arg_pool=set([  i[3] for i in args])
    arg_pool =   "".join(list(arg_pool))
   
    # print(arg_pool)
    
    
    type_pool=set([  ":".join( translations.get(j,j)  for j in   i[0].replace("类型为","").split(":") )  for i in args])
    
    type_pool=  "  ".join(  list(type_pool) )
    return tri_pool,arg_pool,type_pool
    

# font = ImageFont.truetype('C:\Windows\Fonts\simsun.ttc', 20)  # 宋体
# font = ImageFont.truetype(font='C:\Windows\Fonts\STXINGKA.TTF', size=20)   # 楷体
# font = ImageFont.truetype(font='C:\Windows\Fonts\simhei.ttf', size=20)  # 黑体
# font = ImageFont.truetype(font='/home/baoxiaoyi/emnlp2024/font/chinese/FanWunMing/FanWunMing-M.ttf', size=40)  # 繁体
font = ImageFont.truetype(font='../font/chinese/FanWunMing/FanWunMing-M.ttf', size=30)  # 宋体

file_name="zh_train_seq_oneie.json"
task="ee_nl"
f=open("../raw_data/"+file_name,"r")

data=[json.loads(i) for i in f.readlines()]

f.close()


print(data[0])


f=open("../raw_data/generated_predictions.jsonl","r")

data_predict_result=[json.loads(i) for i in f.readlines()]

f.close()




# os.makedirs("./final_oneie_tradition")



f=open("./final_oneie_tradition/"+task+"_"+file_name,"w")

f.write("[\n")
for index,(i,j) in enumerate(zip(data,data_predict_result)):
    image = Image.new('RGB', (500, 500),
                        (255, 255, 255))  # Image.new( mode, size, color ) => image 新建长宽240像素，背景色为（0,0,0）的画布对象
    
    iwidth, iheight = image.size
    draw = ImageDraw.Draw(image)  # 新建画布绘画对象
    prompt = i["input_ee"]
    label = i["target_nl"]
    
    tri_pool, arg_pool, type_pool= build_pool(j["predict"])
    for i in range(len(prompt)):
        # x和y代表元素的位置，图片左上角为(0,0)
        x = (i % 16) * 30  # 每行11个元素，每个元素占20个像素
        y = (i // 16) * 30  # 每列11个元素，每个元素占20个像素
        
        # tt=  S2T(prompt[i]) if is_chinese(prompt[i]) else prompt[i]
        # print(tt)
        
        if prompt[i] in tri_pool:
            draw.text((x, y),S2T(prompt[i]) , (255, 0, 0), font) # red # (255,255,0)为黄色
        elif prompt[i] in arg_pool:
            draw.text((x, y),S2T(prompt[i]) , (0, 255, 0), font) # green # (255,255,0)为黄色
        else:
            draw.text((x, y),S2T(prompt[i]) , (0, 0, 0), font)  # (255,255,0)为黄色
            
    for i in range(len(type_pool)):
        x = (i % 16) * 30
        draw.text((x, 470),S2T(type_pool[i]) , (0, 0, 255), font) # blue # (255,255,0)为黄色
        
        
            
    enh_con = ImageEnhance.Contrast(image)  # 创建一个调整图像对比度的增强对象。增强因子为0.0将产生纯灰色图像；为1.0将保持原始图像。
    image = enh_con.enhance(
        factor=4)  # 该方法返回一个增强过的图像。变量factor是一个浮点数，控制图像的增强程度。变量factor为1将返回原始图像的拷贝；factor值越小，颜色越少（亮度，对比度等）
    image.save("./final_oneie_tradition/"+file_name.split(".")[0]+"_"+task+"_"+str(index)+'.jpg')
    
    img="final_oneie_tradition/"+file_name.split(".")[0]+"_"+task+"_"+str(index)+'.jpg'
    
    
    final=dict()
    
    final["prompt"]="图中包含了句子：'"+prompt+"'的字形字体信息，结合图片和文本内容抽取其中的事件信息。"
    final["label"]=label
    final["img"]=img
    
    json_str=json.dumps(final,ensure_ascii=False)
    
    if index!=len(data)-1:
        f.write(json_str+",\n")
    else:
        f.write(json_str+"\n")
    
f.write("]")
f.close()
    
    
    

