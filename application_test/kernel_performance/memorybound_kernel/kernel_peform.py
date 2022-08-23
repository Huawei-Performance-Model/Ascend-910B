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
ans=[]#本程序最终保存到一个json文件当中去
all_cfg=open("../../../config/all_config.json",encoding="utf-8")
all_config=json.load(all_cfg)
cuda_choice=0   #这里也直接指定使用哪块GPU推理
optype_to_cube=all_config['cube_info']
device = torch.device(cuda_choice if torch.cuda.is_available() else "cpu")#同时在一块GPU上推理
file_size={}
file_size[3]="../../../memorybound_size3/all_norm_tasktime.csv"
file_size[4]="../../../memorybound_size4/all_norm_tasktime.csv"
file_size[5]="../../../memorybound_size5/all_norm_tasktime.csv"
model={}#这里就不讲究了，直接手动挑选负载GPU
model[3]=torch.load("../../../memorybound_size3/size3_best.pth",map_location=device)
model[4]=torch.load("../../../memorybound_size4/size4_best.pth",map_location=device)
model[5]=torch.load("../../../memorybound_size5/size5_best.pth",map_location=device)
operator_cnt3=0
operator_cnt4=0
operator_cnt5=0
loss3=0
loss4=0
loss5=0


#还是需要算子子集
optype_to_opid={}
opid_to_optype={}
operator_subset={}#这几个使用的时候都带上下标就可以了

for i in range(3,6):#暂时只有长度为3，长度为4的模型
    cur_size_pre_training_cfg=open("../../../config/pre_training_config_for_size"+str(i)+".json",encoding="utf-8")
    cur_size_pre_training_config=json.load(cur_size_pre_training_cfg)
    operator_subset[i]=cur_size_pre_training_config["Operator_subset"]
    optype_to_opid[i]=operator_subset[i]
    opid_to_optype[i]={}
    for j in optype_to_opid[i].keys():
        opid_to_optype[i][optype_to_opid[i][j]]=j

Op_loss={}
Op_cnt ={}

for i in range(3,6):
    file=pd.read_csv(file_size[i])
    with open(file_size[i],'r',encoding='utf-8',newline='') as file:
        content = csv.reader(file)
        cnt=0
        for row in content:
            if cnt>=1:
                #从这里开始可以取出label,并进行推理
                OP_id=int(row[0])#####注意这是对应shape下的算子编号
                myname=opid_to_optype[i][OP_id]#这个是真实名字
                label= float(row[3])
                list=row[0:3]+row[4:]
                for x in range(0,len(list)):
                    list[x]=float(list[x])
                input_tensor=[]
                for j in range(0,len(operator_subset[i])):#######################################################################onehot
                    if j==OP_id:
                        input_tensor.append(1)
                    else:
                        input_tensor.append(0)
                for x in range(1,len(list)):
                    input_tensor.append(list[x]) 
                time=model[i](torch.Tensor(input_tensor).unsqueeze(0).to(device)).item()

                if myname in Op_loss.keys():#存在过，直接加上去
                    Op_loss[myname]=Op_loss[myname]+abs(time-label)/label
                    Op_cnt[myname]=Op_cnt[myname]+1
                else:
                    Op_loss[myname]=abs(time-label)/label
                    Op_cnt[myname]=1


                if i==3:
                    loss3=loss3+abs(time-label)/label
                    operator_cnt3=operator_cnt3+1
                elif i==4:
                    loss4=loss4+abs(time-label)/label
                    operator_cnt4=operator_cnt4+1
                elif i==5:
                    loss5=loss5+abs(time-label)/label
                    operator_cnt5=operator_cnt5+1
            cnt=cnt+1
print(operator_cnt3,loss3)
print(operator_cnt4,loss4)
print(operator_cnt5,loss5)

#需要统计的是每一种算子的精度！！！！

for i in Op_loss.keys():
    print(i,Op_loss[i],Op_cnt[i],Op_loss[i]/Op_cnt[i])
    list = [i,Op_loss[i]/Op_cnt[i]]
    data = pd.DataFrame([list])
    data.to_csv('./answer.csv',mode='a',header=False,index=False)#mode设为a