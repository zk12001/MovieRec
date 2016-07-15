# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:10:43 2016

@author: Zhang
"""

import pandas as pd

data = pd.read_csv('dataprocessd.csv')
data_new = data[['uid','vid','rate']]
user_groups = data_new.groupby('uid')
ftr = open('traindata.csv','w+')
fte = open('testdata.csv','w+')
count = 0
for user in user_groups.groups:
    temp = user_groups.get_group(user)
    l = len(temp)/5
    for i in range(len(temp)):
        line = str(user)+','+str(temp.iloc[i]['vid'])+','+str(temp.iloc[i]['rate'])+'\r\n'
        if(i<l):
            fte.write(line)
            count+=1
            print count
        else:
            ftr.write(line)
            count+=1
            print count
ftr.close()
fte.close()
    

