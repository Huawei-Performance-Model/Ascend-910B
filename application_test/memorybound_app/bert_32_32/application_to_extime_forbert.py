#现在这个应用就需要读取多种长度的tensor了
import os
import time
import sqlite3
import csv
import pandas as pd
import argparse
import json
import ast
from fnmatch import fnmatch,fnmatchcase

from unicodedata import name
import torchvision.models as models
import torch
import torch.nn as nn
from net import Net
import math

import sys
sys.path.append("../../../src_code/")
from cal_flops import *
from input_to_tensor import *
from distin_size import *
Model_name=os.path.basename(os.getcwd())
print(Model_name)

ans=[]#本程序最终保存到一个json文件当中去



all_cfg=open("../../../config/all_config.json",encoding="utf-8")
all_config=json.load(all_cfg)
cuda_choice=all_config['testing_cuda_choice']
optype_to_cube=all_config['cube_info']


frame_path=Model_name+".csv"#bert.csv
graph_path="./graph.json"
file = pd.read_csv(frame_path)
g=open(graph_path,encoding="utf-8")


device = torch.device(cuda_choice if torch.cuda.is_available() else "cpu")###############暂时尝试一下，三个模型能同时加载到一个GPU上吗
model={}#这三个模型也都放到字典当中去



#一次性把所有的算子子集以及模型加载进来？然后直接使用cur_size=distin(input_s)的这个下标直接访问
for i in range(3,6):#暂时只有长度为3，长度为4的模型
    model[i] = torch.load("../../../model_size_"+str(i)+"/size"+str(i)+"_best.pth",map_location=device)#肯定需要放到GPU上去测试
    #model[i]=model[i].to(device)



for index, row in file.iterrows():#整个程序就这么一个大循环
    #首先区分这个算子输入哪一种shape模型，再去调用相应的算子子集
    input_s = ast.literal_eval(row['op_info'])
    cur_size=distin(input_s)#获取3,4,5

    if not(cur_size==3 or cur_size==4 or cur_size==5):
        continue

    if not(row['op_type'] in operator_subset[cur_size]):#############################直接不在算子子集里面的直接跳过
        continue

    if(cur_size==5 and row['op_type']=="TransData"):#############################################################################bert专属
        continue



    flops=0#每个算子都需要计算flops
    pre_back=0
    if row['subgraph']=='Gradients':#1表示反向传递loss，权值更新节点单独设置为2
        pre_back=1
    if row['op_type']=='Conv2DBackpropFilter':##############################################这个之后可以改掉
        pre_back=2
    #datatype就使用第一个输入的datatype吧。。
    
    datatype=input_s["input_0"]["data_type"]
    dat=-1
    if datatype=='NUMBER_TYPE_FLOAT':
        dat=0
    elif datatype=='NUMBER_TYPE_FLOAT16':
        dat=1
    ID=optype_to_opid[cur_size][row['op_type']]

    flops,stride=get_flops(row['op_type'],input_s)
    if stride==-1:
        ans.append({'Optype':row['op_type'],'full_op_name':row['full_op_name'],'FLOPS':flops,'Dataset':pre_back,'datatype':dat,'id':ID,'opinfo':row['op_info']})
    else:
        ans.append({'Optype':row['op_type'],'full_op_name':row['full_op_name'],'FLOPS':flops,'stride':stride,'Dataset':pre_back,'datatype':dat,'id':ID,'opinfo':row['op_info']}) 
    




#建立一个映射关系吧，那就是fullopname到前向节点集，以及后向节点集的映射关系
full_opname_to_pre_node={}
full_opname_to_succ_node={}
#---------------------using  subgraph.txt---------------------------------------------
h=open("subgraph.txt")
line=h.readline()
while line:
#---------------------------此时是处理名字-------------------------------------
    my_list1=line[:-1].split(" ")
    GRAPH_ID=my_list1[0]
    FULL_NAME=my_list1[1]#使用这个名称
#---------------------------此时是处理前向节点---------------------------------
    line=h.readline()
    my_list2=line[:-1].split(" ")
    pre_node_num=my_list2[0]
    list_temp=[]
    for i in range(1,int(pre_node_num)+1):
        list_temp.append(my_list2[i])
    full_opname_to_pre_node[FULL_NAME]=list_temp
#---------------------------此时是处理后继节点------------------------------------------
    line=h.readline()
    my_list3=line[:-1].split(" ")
    succ_node_num=my_list3[0]
    list_temp=[]
    for i in range(1,int(succ_node_num)+1):
        list_temp.append(my_list3[i])
    full_opname_to_succ_node[FULL_NAME]=list_temp
    line=h.readline()
h.close()
full_opname_to_graphid={}
graph=json.load(g)
node=graph['graph']['node']
for i in node:
    full_opname_to_graphid[i['fullName']]=i['name']
g.close()
#然后在json文件加上这些信息：
for i in ans:
    i['subgraph_id']=full_opname_to_graphid[i["full_op_name"]]
    i['pre_node']=full_opname_to_pre_node[i["full_op_name"]]
    i['succ_node']=full_opname_to_succ_node[i["full_op_name"]]
    



fullname_to_extime={}#直接使用profiling表格就能得到每个算子的执行时间
file_time=pd.read_csv("./aicore_intermediate_0_detail.csv")
for index, row in file_time.iterrows():
    fullname_to_extime[row['full_op_name']]=row['execution_time']*1000



total_real=0
total_predict=0

#这个程序需要大修啊
for i in ans:
    input_s = ast.literal_eval(i['opinfo'])
    cur_size=distin(input_s)

    input_tensor=input_cast(input_s,i["id"],i["datatype"],i["Dataset"],i["FLOPS"],i["Optype"],optype_to_cube[i["Optype"]],cur_size)
    
    time=model[cur_size](torch.Tensor(input_tensor).unsqueeze(0).to(device)).item()
    i['extime']=fullname_to_extime[i['full_op_name']]
    i['procast']=time
    total_real=total_real+i['extime']
    total_predict=total_predict+i['procast']

    i['cur_size']=cur_size################################





runtimedb_path='./runtime.db'
conn = sqlite3.connect (runtimedb_path)
c=conn.cursor()
cursor=c.execute("SELECT vec_ratio,mac_ratio,task_id,stream_id,mte1_ratio,mte2_ratio,mte3_ratio from MetricSummary")
total_memorybound={}
total_memorybound_times={}
for row in cursor:
    id=(row[2],row[3])#(task_id ,stream_id)
    if id in total_memorybound:#以前出现过
        total_memorybound[id]=total_memorybound[id]+row[5]/max(row[0],row[1])
        total_memorybound_times[id]=total_memorybound_times[id]+1
    else:#第一次出现过
        total_memorybound[id]=row[5]/max(row[0],row[1])
        total_memorybound_times[id]=1
#这样就建立了二元映射到算子执行时间的映射关系
#通过列表建立fullname到这个的映射关系：
fullname_to_memorybound={}
fullname_to_memorybound_times={}
file = pd.read_csv(frame_path)
for index, row in file.iterrows():
    input_s = ast.literal_eval(row['op_info'])
    cur_size=distin(input_s)#获取3,4,5
    if not(cur_size==3 or cur_size==4 or cur_size==5):
        continue
    if not(row['op_type'] in operator_subset[cur_size]):#############################直接不在算子子集里面的直接跳过
        continue
    id=(row['task_id'],row['stream_id'])
    fullname_to_memorybound[row['full_op_name']]=total_memorybound[id]
    fullname_to_memorybound_times[row['full_op_name']]=total_memorybound_times[id]
    fullname_to_memorybound[row['full_op_name']]=fullname_to_memorybound[row['full_op_name']]*1.0/fullname_to_memorybound_times[row['full_op_name']]
#映射关系建立好啦



model_memorybound={}#这三个模型也都放到字典当中去
for i in range(3,6):#暂时只有长度为3，长度为4的模型
    model_memorybound[i] = torch.load("../../../memorybound_size"+str(i)+"/size"+str(i)+"_best.pth",map_location=device)#肯定需要放到GPU上去测试


real_ratio=0
predict_ratio=0

cntxx=0
cntxxx=0
cntreal=0
for i in ans:
    input_s = ast.literal_eval(i['opinfo'])
    cur_size=distin(input_s)
    input_tensor=input_cast(input_s,i["id"],i["datatype"],i["Dataset"],i["FLOPS"],i["Optype"],optype_to_cube[i["Optype"]],cur_size)
    memorybound=model_memorybound[cur_size](torch.Tensor(input_tensor).unsqueeze(0).to(device)).item()
    i['ex_memorybound']=fullname_to_memorybound[i['full_op_name']]
    i['procast_memorybound']=memorybound

    real_ratio=real_ratio+fullname_to_memorybound[i['full_op_name']]*i["extime"]/total_real

    predict_ratio=predict_ratio+memorybound*i["procast"]/total_predict
    cntxx=cntxx+1
    cntxxx=cntxxx+memorybound
    cntreal=cntreal+fullname_to_memorybound[i['full_op_name']]


print("real:")
print(real_ratio)
print("predict:")
print(predict_ratio)

