import numpy as np
from collections import Counter
from feature_extraction import *
import pickle
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7255, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ml_model():
    def __init__(self):
        self.model_xgb = pickle.load(open('model/XGB_model.pickle','rb'))
        self.model_lgb = pickle.load(open('model/LGB_model.pickle','rb'))
        self.model_rfc = pickle.load(open('model/RF_model.pickle','rb'))
        self.PTH_model = Net().to(device)
        self.PTH_model.load_state_dict(torch.load('model/PTH_model.pickle',map_location='cpu'))
        self.PTH_scaler = pickle.load(open('model/PTH_scaler.pickle','rb'))
        self.modle_stk = pickle.load(open('model/ST_model.pickle','rb'))
        self.keywords = np.load('keywords_array.npy',allow_pickle=True)
    def get_asm_features(self,asm_file_path):
        #with open(asm_file_path,'r',encoding='UTF-8') as f:
        with open(asm_file_path,'r',encoding='UTF-8',errors='ignore') as f:
            f.seek(0)
            data_defines = asm_data_define(f)
            f.seek(0)
            meta_data = asm_meta_data(asm_file_path, f)
            f.seek(0)
            opcodes = asm_opcodes(f)
            f.seek(0)
            registers = asm_registers(f)    
        return data_defines + meta_data + opcodes + registers
    def get_bytes_features(self,bytes_file_path):
        with open(bytes_file_path,'r',errors='ignore') as f:
            f.seek(0)
            entropy = byte_entropy(f)
            f.seek(0)
            image1 = byte_image1(f)
            f.seek(0)
            meta_data = byte_meta_data(bytes_file_path, f)
            f.seek(0)
            oneg = byte_1gram(f)
            f.seek(0)
            str_lengths = byte_string_lengths(f)
        return entropy + image1 + meta_data + oneg +str_lengths
    def get_keyword_counts(self,asm_file_path, keywords):
        with open(asm_file_path,'r',encoding='UTF-8',errors='ignore') as f:
            lines = f.readlines()
            words = list()
            for l in lines:
                words += l.split()
            word_Counter = Counter(words)
            keyword_counts = [word_Counter[kw] for kw in keywords]
        return keyword_counts
    def preprocess(self,asm_file_path, bytes_file_path):
        asm_features = self.get_asm_features(asm_file_path)
        bytes_features = self.get_bytes_features(bytes_file_path)
        kw_features = self.get_keyword_counts(asm_file_path,self.keywords)
        return asm_features+bytes_features+kw_features
    def predict_proba(self,features):
        pred_prob_xgb = self.model_xgb.predict_proba([features])[0]
        pred_prob_lgb = self.model_lgb.predict_proba([features])[0]
        pred_prob_rfc = self.model_rfc.predict_proba([features])[0]
        pth_feature = self.PTH_scaler.transform([features])
        tensor_features = torch.tensor([pth_feature]).float().to(device)
        pred_prob_pth = F.softmax(self.PTH_model(tensor_features)[0],dim=1).cpu().data.numpy()[0]        
        stk_prob = np.concatenate((pred_prob_xgb,pred_prob_lgb,pred_prob_rfc,pred_prob_pth))
        pred_prob_stk = self.modle_stk.predict_proba([stk_prob])[0]
        return pred_prob_stk
    def get_labels(self):
        return ['Trojan-Ransom.Win32.Blocker.xxx','Trojan-Ransom.Win32.GandCrypt.xxx','Trojan-Ransom.Win32.Foreign.xxx','Trojan-Ransom.Win32.PornoBlocker.xxx','Trojan-Ransom.Win32.Wanna.xxx','HEUR:Trojan-Ransom.xxx.Blocker.gen','HEUR:Trojan-Ransom.xxx.Gen.gen','Trojan-Ransom.PHP.xxx','HEUR:Trojan-Ransom.Win32.xxx.vho','others']
    def file_to_result_proba(self,asm_file_path, bytes_file_path):
        features = self.preprocess(asm_file_path, bytes_file_path)
        result_proba = self.predict_proba(features)
        return result_proba