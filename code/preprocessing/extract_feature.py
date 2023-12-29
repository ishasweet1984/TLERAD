#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ML_model import *
import os
import pandas as pd


# In[ ]:


bytesdir = 'byte/'   # bytefiles dir
asmdir = 'asm/'      # asm files dir
sampledir = 'bin/'   # binary    dir


# In[ ]:


samplenamelist = []
samplenamelist = os.listdir(sampledir)
print(len(samplenamelist))


# In[ ]:


ML = ml_model()


# In[ ]:


# this ceil is used to generate feature file
# our original binary file name from VT is like <md5_value>.danger, please modify code accorrding to your filename
errcount = 0
successcount = 0                                   # record analysis success amount
recordlist = []                                    # record analysis result of each sample
errfile = open("errordir/err_VT_train_bytelist.txt",'a')  # file uses to record error msg
for name in samplenamelist:                        
    orifilename = name.split(".")[0]               # name = xxx.danger, orifilename = xxx
    bytes_file = bytesdir+orifilename+".byte"      # bytes_file = <bytedir>/xxx.byte
    asm_file = asmdir+orifilename+".asm"           # asm_file = <asmdir>/xxx.asm
    bin_file = sampledir+name                      # bin_file = <bindir>/xxx.danger
    print("processing sample="+name)
    if os.path.isfile(bytes_file) and os.path.isfile(asm_file):    # both asm file and byte file of binary exist
        analysiscount += 1
        print(analysiscount,"-th sample")
        try:
            preresult = ML.preprocess(asm_file,bytes_file) # generate features
        except:
            print("Error! A problem occurs in sample="+name)
            errfile.write(bytesdir+orifilename+".byte"+"\n")
            errfile.flush()
        else:
            if len(preresult) == 7255:             # only the extracted feature amount satisfy the specific number, the process success
                tmplist=[]            
                tmplist.append(name)
                for value in preresult:
                    tmplist.append(value)
                recordlist.append(tmplist)
                successcount += 1
                print(successcount," --th success sample")
            else:
                with open("errordir/err_VT_training_feature.txt",'a') as errorPE:
                    errorPE.write(orifilename+" has error in PEfile analysis"+"\n")
                    errcount += 1


# In[ ]:


# record feature into file
errfile.close()
outfile = open('features/ML_VT_train_feature.txt','a',encoding='UTF-8')  # output feature record file
print("success PE analysis amount:",len(recordlist))
print("pe_tool_error amount:",errcount)
for record in recordlist:
    tmpstr = ""
    for value in range(0,len(record)):
        if value == 0:
            tmpstr = record[value].split(".")[0]    # exclude extname and left md5_value
        elif not value == len(record)-1:
            tmpstr += ","+str(record[value])
        else:
            tmpstr += ","+str(record[value])+"\n"
    outfile.write(tmpstr)
outfile.close()
print("feature file generate finish")


# In[ ]:


# normal prediction process cell samplecode(for loop directory, not single sample)
# If you only want to generate feature file, do not execute and modify this cell
"""
errorlog = open('errordir/ML_testing_err.txt','a',encoding='UTF-8')         # errlog dir
for name in samplenamelist:                  
    orifilename = name.split(".")[0]               # name = xxx.danger, orifilename = xxx
    bytes_file = bytesdir+orifilename+".byte"      # bytes_file = <bytedir>/xxx.byte
    asm_file = asmdir+orifilename+".asm"           # asm_file = <asmdir>/xxx.asm
    bin_file = sampledir+name                      # bin_file = <bindir>/xxx.danger
    print("processing sample="+name)
    if os.path.isfile(bytes_file) and os.path.isfile(asm_file):    # both asm file and byte file of binary exist
        analysiscount += 1
        print(analysiscount,"-th sample")
        try:
            preresult = ML.preprocess(asm_file,bytes_file)         # analysis result in each loop
        except:
            print("Error! A problem occurs in sample="+name)
            errorlog.write(orifilename+" occurs error in processing ML prediction"+"\n")
            errorlog.flush()
        else:
            print(orifilename+"=",len(preresult))
errorlog.close()
"""

# In[ ]:




