# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#to get count of each disease in a csv
import csv
import collections

disease=collections.Counter()
#file_1="/home/praharsha/Desktop/Nvidia/chestXray_dataset/data.csv"
file_1="/home/praharsha/Desktop/model_14/data_vm.csv"

with open(file_1) as input_file:
    dis=[]
    for row in csv.reader(input_file,delimiter=','):
        count=row[1].count("|") + 1
        dis.extend(row[1].split("|"))
labels = ["Cardiomegaly","Emphysema","Effusion","Hernia","No Finding","Infiltration","Mass","Nodule","Pneumothorax","Pleural_Thickening","Atelectasis","Fibrosis","Edema","Consolidation","Pneumonia"]   
al=dict()
for i in range(15):
    al[labels[i]]=dis.count(labels[i])
    
al_sorted=(sorted(al.items(),key=lambda x: -x[1]))
for i in al_sorted:
    print i 


print sorted(al.items(),key=lambda x: -x[1])
print sorted(al.items(), key=lambda x: (-x[1], x[0]))




# to prepare a csv which only contains selected diseases.

req=["Infiltration","Effusion","Atelectasis","Nodule","Mass"]


with open('/home/praharsha/Desktop/model_14/data_vm_5.csv', 'wb') as f: # output csv file
    writer = csv.writer(f)
    with open(file_1) as input_file:
        for row in csv.reader(input_file,delimiter=','):
            lab=row[1].split("|")
            #print lab
            dis=[]
            for i in lab:
                req=set(req)
                if i in req:
                    #print i
                    dis.append(i)
                    #print dis
            if len(dis)>0:
                lab1="|"
                lab1=lab1.join(dis)
                row[1]=lab1    
                writer.writerow(row)
            else:
                lab1="ignore"
                row[1]=lab1
                writer.writerow(row)
            
            #print row[1]
    
    
# to remove all unwanted diseases from csv
import csv

my_file_name = "/home/praharsha/Desktop/model_14/data_vm_5.csv"
cleaned_file_name = "/home/praharsha/Desktop/model_14/data_clean.csv"
ONE_COLUMN = 1
remove_words = ['ignore']
with open(my_file_name, 'r') as infile, open(cleaned_file_name, 'wb') as outfile:
    writer = csv.writer(outfile)
    for row in csv.reader(infile, delimiter=','):
        column = row[ONE_COLUMN]
        if not any(remove_word in column for remove_word in remove_words):
            writer.writerow(row)
