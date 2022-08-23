'''
to distinguish input_size,for corresponding trained deep learning model
'''
import os
import time
import sqlite3
import csv
import pandas as pd
import argparse
import json
import ast
from fnmatch import fnmatch,fnmatchcase


def getlen(str):
    if str=="":
        return 1
    str=str.split(",")
    ret=0
    for i in str:
        ret=ret+1
    return ret
def distin(input_s):#目前的版本是获取输入的单个输入的最大维度
    max_size=0
    for i in input_s.keys():
        if(fnmatch(i,"input_*") or fnmatch(i,"output_*")):
            max_size=max(max_size,getlen(input_s[i]["shape"]))
    return max_size


