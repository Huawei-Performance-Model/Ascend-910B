import os
import time
import sqlite3
import csv
import pandas as pd
import argparse
import json
import ast
from fnmatch import fnmatch,fnmatchcase
import math

max_dimension_for_all_operator={}
max_input_output_num={}
mean_i={}
std_i={}
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
    max_dimension_for_all_operator[i]=cur_size_pre_training_config["max_dimension_for_all_operator"]
    max_input_output_num[i]=cur_size_pre_training_config["max_input_output_num"]


    cur_size_std_cfg=open("../../../config/std_config_for_size"+str(i)+".json",encoding="utf-8")
    cur_size_std_config=json.load(cur_size_std_cfg)
    mean_i[i]=cur_size_std_config["mean_dic"]
    std_i[i]=cur_size_std_config["std_dic"]
    


def my_list_max_dimension_for_all_operator(str,cursize):#返回长度为max_dimension_for_all_operator的列表,通常为3，4，5，6
    max_dim=max_dimension_for_all_operator[cursize]
    if str=="":#it means this is a scalar
        ret_list=[1]
        for i in range(0,max_dim-1):
            ret_list.append(0)
        return ret_list#return [1,0,0,0,0,0]
    str=str.split(",")
    ret_list=[]
    cnt=0
    for i in str:
        ret_list.append(i)
        cnt=cnt+1
    for i in range(0,max_dim-cnt):
        ret_list.append(0)#剩余位置补0
    return ret_list



def input_cast(input_s,ID,datatype,dataset,flops,optype,cube,cur_size):#########################原始方法，没有用flops的onehot
    
    input_tensor=[]
    dat=0.0
    dataset=0.0
    if std_i[cur_size]['1']==0:
        dat=0.0
    else:
        dat=(datatype-mean_i[cur_size]['1'])/std_i[cur_size]['1']
    if std_i[cur_size]['2']==0:
        dataset=0.0
    else:
        dataset=(dataset-mean_i[cur_size]['2'])/std_i[cur_size]['2']


    flops=math.log(flops)
    for j in range(0,len(operator_subset[cur_size])):#######################################################################onehot
        if j==ID:
            input_tensor.append(1)
        else:
            input_tensor.append(0)
    input_tensor.append(dat)
    input_tensor.append(dataset)
    input_tensor.append(flops)
    input_tensor.append(cube)
    #############################################################
    #需要使用input_s来得到相关的剩下的14*6个输入
    row_list=[]
    cur_cnt=0

    if optype=="AddN" or optype=="Mul":#无论多长，只放入2个输入,一个输出
        curlist=my_list_max_dimension_for_all_operator(input_s["input_0"]["shape"],cur_size)#利用这个东西直接得到一个长度为6的列表
        for k in curlist:
            row_list.append(float(k))
            cur_cnt=cur_cnt+1
        curlist=my_list_max_dimension_for_all_operator(input_s["input_1"]["shape"],cur_size)#利用这个东西直接得到一个长度为6的列表
        for k in curlist:
            row_list.append(float(k))
            cur_cnt=cur_cnt+1
        curlist=my_list_max_dimension_for_all_operator(input_s["output_0"]["shape"],cur_size)#利用这个东西直接得到一个长度为6的列表
        for k in curlist:
            row_list.append(float(k))
            cur_cnt=cur_cnt+1
    else:
        for j in input_s.keys():
            if(fnmatch(j,"input_*") or fnmatch(j,"output_*")  ):
                curlist=my_list_max_dimension_for_all_operator(input_s[j]["shape"],cur_size)#利用这个东西直接得到一个长度为6的列表
                for k in curlist:
                    row_list.append(float(k))
                    cur_cnt=cur_cnt+1

    max_shape_len=max_input_output_num[cur_size]*max_dimension_for_all_operator[cur_size]#比如说这个东西的长度是14*6=84


    for j in range(cur_cnt,max_shape_len):
        row_list.append(0.0)
    ###############################然后把这max_shape_len个数字正则化之后输入input_tensor当中
    for j in range(0,max_shape_len):#需要遍历0到max_shape_len-1
        x=row_list[j]          #从6开始
        if(std_i[cur_size][str(j+6)]==0):###################对于那些全都相等的列，直接置为0
            input_tensor.append(0.0)
            continue
        x=(x-mean_i[cur_size][str(j+6)])/std_i[cur_size][str(j+6)]
        input_tensor.append(x)
    return input_tensor




'''
def input_cast(input_s,ID,datatype,dataset,flops,optype):#没有std的版本，尝试对于flops做四次根号处理
    input_tensor=[]
    dat=datatype
    #flops=(flops**0.5)**0.5

    flops=flops** (1 / 3.0)

    #flops=math.log(flops)


    cube=optype_to_cube[optype]
    for j in range(0,len(operator_subset)):#######################################################################onehot
        if j==ID:
            input_tensor.append(1)
        else:
            input_tensor.append(0)
    input_tensor.append(dat)
    input_tensor.append(dataset)
    input_tensor.append(flops)
    input_tensor.append(cube)
    #############################################################
    #需要使用input_s来得到相关的剩下的14*6个输入
    cur_cnt=0
    for j in input_s.keys():
        if(fnmatch(j,"input_*") or fnmatch(j,"output_*")  ):
            curlist=my_list6(input_s[j]["shape"])#利用这个东西直接得到一个长度为6的列表
            for k in curlist:
                input_tensor.append(float(k))
                cur_cnt=cur_cnt+1

    for j in range(cur_cnt,84):
        input_tensor.append(0.0)
    return input_tensor
'''

