'''
这个应该是第二步
应该需要筛选出算子子集吧。。改写一下这个程序
'''

import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from fnmatch import fnmatch,fnmatchcase
import sqlite3
import ast



Model_name=os.path.basename(os.getcwd())

graph_path="./graph.json"


f=open(graph_path,encoding="utf-8")
graph=json.load(f)
#=open('test.txt','w')
h=open('fullgraph.txt','w')
node=graph['graph']['node']


for i in node:
    h.write(str(i['name']))
    h.write('\t')
    h.write(i['fullName'])
    h.write('\n')
    if 'input' in i:
        cnt=0
        for j in i['input']:
            if cnt==0:
                h.write(str(j['name']))
            else:
                h.write(' '+str(j['name']))
            cnt=cnt+1
        h.write('\n')
#首先到这一步为止只是获取了计算图当中所有算子的前向信息
#我们还需要知道哪些算子是需要被筛选的，也就是需要确定一下算子子集
#实现方法就是在最后一行加上被选中的算子，其在计算图当中的id




#首先这个输入framework，然后建立一个图-fullname-framework表格-inputs
#然后就可以借用之前的方法来确定哪些算子需要被筛选了。

fullname_to_inputs={}
frame_path=Model_name+".csv"#bert.csv
file = pd.read_csv(frame_path)
for index, row in file.iterrows():
    fullname_to_inputs[row['full_op_name']]=ast.literal_eval(row['op_info'])



import sys
sys.path.append("../../../src_code/")
from input_to_tensor import *
from distin_size import *


all_cfg=open("../../../config/all_config.json",encoding="utf-8")
all_config=json.load(all_cfg)
optype_to_cube=all_config['cube_info']



bnt=0
for i in node:
    #判断一下这个算子的optype是否在算子子集当，如果是的话就加上
    if not(i["opType"] in optype_to_cube):#第一轮筛选
        continue
    input_s=fullname_to_inputs[i["fullName"]]
    cur_size=distin(input_s)#获取3,4,5
    if not(cur_size==3 or cur_size==4 or cur_size==5):
        continue
    if not(i["opType"] in operator_subset[cur_size]):#############################直接不在算子子集里面的直接跳过
        continue
    if bnt==0:
        h.write(str(i['name']))
    else:
        h.write(' '+str(i['name']))
    bnt=bnt+1


f.close()
h.close()