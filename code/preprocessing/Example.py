#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys,os
from ML_model import *
bytes_path = sys.argv[1]
asm_path = sys.argv[2]



# In[2]:


bytes_file = 'files/test.bytes'  # default
asm_file = 'files/test.asm'      # default
if os.path.isfile(bytes_path):
    bytes_file = bytes_path
if os.path.isfile(asm_path):
    asm_file = asm_path


# Class

# In[3]:


ML = ml_model()


# features

# In[4]:


# this cell can get print array of extracted features
ML.preprocess(asm_file,bytes_file)




# In[5]:


# this cell can acquire predict result and save in ndary 
ndary = ML.file_to_result_proba(asm_file,bytes_file)


# In[6]:


print(ndary)
return ndary

# In[7]:


#samplescorelist = []
#for score in ndary:
#    samplescorelist.append(score)
#print(samplescorelist)
#max = np.amax(ndary)
#print(max)


# # 取得預測類別表

# In[8]:


#ML.get_labels()


# In[ ]:




