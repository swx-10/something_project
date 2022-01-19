import os,time,pickle
from keras import Input
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import class_distribution
from preprocess import label2id
from run import load_bert,reverse_model,predict,model_name
from sklearn.metrics import classification_report
import logging as log

log_path = os.path.join(os.path.dirname(__file__),'output.log')
log.basicConfig(filename=log_path,level=log.DEBUG)

each_class_samples = 100
class_num = 5

tokenizer = load_bert(just_load_token=True)
save_path = os.path.join(os.path.dirname(__file__),model_name)
model = reverse_model(save_path)


data_path = os.path.join(os.path.dirname(__file__),'data','data_new.pkl')
text = pickle.load(open(data_path,'rb+'))
np.random.shuffle(text)
text = pd.DataFrame(text,columns=['lb',"tx"])

def process(text,each_class_samples,class_num):
    y_true,sentences = [],[]
    for i in range(class_num):
        a = text[text.lb==i][:each_class_samples]
        for j in a.tx.values:
            sentences.append(j)
        for c in a.lb.values:
            y_true.append(c)

    content = []
    for x in sentences:
        if len(x)>510:
            x = x[:127]+x[-381:]
        content.append(x)
    return y_true,content

if __name__ == "__main__":
    while True:
        y_true,content = process(text,each_class_samples,class_num)

        st = time.time()
        print("开始预测")
        label = predict(content,model,tokenizer)
        pt = time.time()
        label = [label2id.get(i) for i in label]

        target_names = ['非常不满-服务敏感','非常不满-费用敏感','非常不满-渠道敏感','一般不满','比较不满']
        log.info(classification_report(y_true,label,target_names=target_names))
        log.info(f'共计{each_class_samples*class_num}个样本,单个样本预测耗时{(pt - st)/each_class_samples*class_num}s')

        choice = input('是否继续(y/n)').lower()
        if choice == 'n':
            break
