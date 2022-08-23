import os
import time
import sqlite3
import csv
import pandas as pd
import argparse
import json
import ast
from fnmatch import fnmatch,fnmatchcase



def my_mul(str):#能够处理各种类型的"32,1,224,224,16"或者是""
    if str=="":
        return 1
    str=str.split(",")
    ret=1
    for i in str:
        ret=ret*int(i)
    return ret
def addall(input_s):#这个的处理方法就是把所有的算子的tensor，内部乘，外部加，全部处理。
    flops=0
    for i in input_s.keys():
        if(fnmatch(i,"input_*")):#如果是input的话
            flops=flops+my_mul(input_s[i]["shape"])
    return flops
def mulall(input_s):#这个的处理方法就是把所有的算子的tensor，全部都乘起来
    flops=1
    for i in input_s.keys():
        if(fnmatch(i,"input_*")):#如果是input的话
            flops=flops*my_mul(input_s[i]["shape"])
    return flops
def getstride(s1,s2):#使用s1的 "2,1,768,1280,16"和s2的"2,4,384,640,16"得到stride
    s1=s1.split(",")
    s2=s2.split(",")
    return [(float(s1[2])/float(s2[2])),(float(s1[3])/float(s2[3]))]
def getmat_flops(input_s):
    input0=input_s["input_0"]["shape"].split(",")
    input1=input_s["input_1"]["shape"].split(",")
    output0=input_s["output_0"]["shape"].split(",")
    a=input0[0]
    b=input0[1]
    c=input1[0]
    d=input1[1]
    e=output0[0]
    f=output0[1]

    a=int(a)
    b=int(b)
    c=int(c)
    d=int(d)
    e=int(e)
    f=int(f)

    res=0
    if c==a:
        if d==b:#有可能二二相等，也有可能四个全相等
            if c==d:
                res=a*b*c
            else:#二二相等，看看输出的结果吧
                if a==e:#说明b，和d是矩阵之间相等量
                    res=a*e*b
                else:#说明a，c是矩阵之间相等量
                    res=b*d*a
        else:#x,b,x,d 有可能是 x,y,x,z有可能是  x,y,x,x，也有可能是 x,x,x,y
            res=a*b*d
    elif c==b:#已经是 a,b,c,d 为x y y d了
        if d==a:#又是二二相等 x y y x
            if a==e:
                res=a*b*d
            else:
                res=a*b*c
        else:#有可能x,y,y,y 或者x y y z
            res=a*b*d
    elif d==a:#已经是 a,b,c,d为  x,z,y,x 或者x,x,y,x
        res=a*b*c
    else:#只有可能d==b了，并且d!=a ,c也不相等
        res=a*c*d
    if input_s["input_0"]["format"]=="FRACTAL_NZ":#注意cube乘以16*16
        res=res*16*16*16
        return res
    else:
        return res

    
def get_flops(optype,input_s):#
    #写一个接口来处理特定算子的flops
    #我们这个程序就是这样运行的：手动输入target算子名，输入是数据库csv文件，然后处理出算子特征工程+执行时间+MTE信息
    if optype=='AddN':
        return addall(input_s),-1
    elif optype=='BNInfer':
        return addall(input_s),-1
    elif optype=='BNTrainingUpdate':
        return addall(input_s),-1
    elif optype=='BNTrainingReduceGrad':
        return addall(input_s),-1
    elif optype=='BNTrainingUpdateGrad':
        return addall(input_s),-1
    elif optype=='Conv2D':#这里考虑了stride对于计算量的影响
        flops=mulall(input_s)
        s=getstride(input_s["input_0"]["shape"],input_s["output_0"]["shape"])
        return flops/s[0]/s[1],s
    elif optype=='Conv2DBackpropFilter':
        flops=mulall(input_s)
        s=getstride(input_s["input_1"]["shape"],input_s["input_0"]["shape"])
        return flops/s[0]/s[1] ,s
    elif optype=='Conv2DBackpropInput':
        flops=mulall(input_s)
        s=getstride(input_s["output_0"]["shape"],input_s["input_0"]["shape"])
        return flops/s[0]/s[1] ,s 
    elif optype=='FusionOp_BNTrainingUpdate_Add_ReLUV2':
        return addall(input_s)+my_mul(input_s["input_0"]["shape"]),-1#再加一次input0的乘法flops
    elif optype=='FusionOp_BNTrainingUpdate_ReLUV2':
        return addall(input_s)+my_mul(input_s["input_0"]["shape"]),-1
    elif optype=='FusionOp_Conv2D_BNTrainingReduce':#这里考虑了stride对于计算量的影响
        s=getstride(input_s["input_0"]["shape"],input_s["output_0"]["shape"])
        return mulall(input_s)/s[0]/s[1] + my_mul(input_s["output_0"]["shape"]),s #计算量就是输入全部乘法，加上第一个输出的乘法
    elif optype=='FusionOp_Conv2DBackpropInput_AddN_ReluGradV2':
        s=getstride(input_s["output_0"]["shape"],input_s["input_0"]["shape"])
        return my_mul(input_s["input_0"]["shape"])*my_mul(input_s["input_1"]["shape"])/s[0]/s[1]+my_mul(input_s["input_2"]["shape"])*2  ,s
    elif optype=='FusionOp_Conv2DBackpropInput_ReluGradV2':
        s=getstride(input_s["output_0"]["shape"],input_s["input_0"]["shape"])
        return my_mul(input_s["input_0"]["shape"])*my_mul(input_s["input_1"]["shape"])/s[0]/s[1]+my_mul(input_s["output_0"]["shape"])  ,s
    elif optype=='FusionOp_MatMul_Mul':
        return getmat_flops(input_s),-1
        #return my_mul(input_s["input_0"]["shape"])*my_mul(input_s["input_1"]["shape"]) ,-1
    elif optype=='MatMul':

        return getmat_flops(input_s),-1

        #return my_mul(input_s["input_0"]["shape"])*my_mul(input_s["input_1"]["shape"]),-1
    elif optype=='MaxPoolGradWithArgmax':
        s=getstride(input_s["output_0"]["shape"],input_s["input_0"]["shape"])
        return addall(input_s) ,s
    elif optype=='MaxPoolWithArgmax':
        s=getstride(input_s["input_0"]["shape"],input_s["output_0"]["shape"])
        return addall(input_s) ,s
    elif optype=='Mul':
        return my_mul(input_s["output_0"]["shape"])*2 ,-1
    elif optype=='ReluGradV2':
        return addall(input_s),-1
    elif optype=='SoftmaxCrossEntropyWithLogits':#目前还没处理这个算子
        return addall(input_s),-1
    elif optype=='TransData':
        return addall(input_s),-1
    elif optype=='Transpose':
        return addall(input_s),-1
    elif optype=='FusionOp_MatMul_GeLU':
        return getmat_flops(input_s),-1
        #return my_mul(input_s["input_0"]["shape"])*my_mul(input_s["input_1"]["shape"]),-1
    elif optype=='FusionOp_MatMul_GeLUGrad':
        return getmat_flops(input_s),-1
        #return my_mul(input_s["input_0"]["shape"])*my_mul(input_s["input_1"]["shape"]),-1
    elif optype=='LayerNorm':
        return addall(input_s),-1
    elif optype=='LayerNormXBackpropV2':
        return addall(input_s),-1
    elif optype=='GeLU':
        return addall(input_s),-1
    elif optype=='GeLUGrad':
        return addall(input_s),-1
    else:#异常情况
        return -1 , -1





