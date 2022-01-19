import json,re,math,os,pickle
import pandas as pd
import numpy as np
import logging as log
from keras_bert.loader import load_vocabulary
from keras_bert.util import get_checkpoint_paths
from keras_bert.bert import TOKEN_CLS,TOKEN_SEP
from keras_bert import Tokenizer
from keras.utils.np_utils import to_categorical
from keras_bert.datasets import get_pretrained,PretrainedList


label2id = {
    '非常不满-服务敏感':0,   # 0.2
    '非常不满-费用敏感':1,   # 0.3
    '非常不满-渠道敏感':2,   # 0.3
    '一般不满':3,           # 0.1 
    '比较不满':4            # 0.1
}

def train_valid_split(data,train_path,valid_path,split_rate=0.9):
    '''
    - data:numpy.array
    - train_path:train data save target
    - valid_path:valid data save target
    - split_rate:train valid split rate
    '''
    np.random.shuffle(data)
    sum = len(data)
    train_data = data[:int(sum*split_rate)]
    valid_data = data[int(sum*split_rate):]
    # save as pickle
    with open(train_path,'wb') as f_t:
        pickle.dump(train_data,f_t)
        log.info('train_data save success!')
        f_t.close()
    with open(valid_path,'wb') as f_v:
        pickle.dump(valid_data,f_v)
        log.info('valid_data save success!')
        f_v.close()
    return len(train_data),len(valid_data)

# def pre_process(data):
#     '''
#     - data:pandas.dataframe
#     - return:numpy.array, cleaned data
#     '''
#     data = data[['情绪等级标注结果','受理内容']].dropna()
#     data['情绪等级标注结果'] = data['情绪等级标注结果'].apply(lambda x:label2id.get(x))
#     data['受理内容'] = data['受理内容'].apply(lambda x:re.sub('[0-9][0-9]{5,}|[\\n]|[/\\r]|[【]|[】]|[A-Za-z0-9_]{5,}|[,]','',x))
#     data['受理内容'] = data['受理内容'].apply(lambda x:re.sub('([0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]{1}|[0-9]{1}[1-9][0-9]{2}|[1-9][0-9]{3})-(((0[13578]|1[02])-(0[1-9]|[12][0-9]|3[01]))|((0[469]|11)-(0[1-9]|[12][0-9]|30))|(02-(0[1-9]|[1][0-9]|2[0-8])))|[0-9]{4}[年][0-9]{1,}[月][0-9]{1,}[日]|[0-9]{1,}[:0-9]{1,}[:0-9]{1,}','',x))
#     # bert allow max_len == 512 包括[CLS][SEP]
#     data['受理内容'] = data['受理内容'].apply(lambda x:x[:510] if len(x)>510 else x)
#     return np.asarray(data)

def pre_process(df_data):
    df_data.columns = ["label","content"]
    data = df_data.copy().dropna()
    # data.columns=["label","content"]
    # 三个专有词表
    data0 = np.asarray(pd.read_csv('./data/服务敏感.csv')).squeeze()
    data1 = np.asarray(pd.read_csv('./data/费用敏感.csv')).squeeze()
    data2 = np.asarray(pd.read_csv('./data/渠道敏感.csv')).squeeze()
    data['label'] = data['label'].apply(lambda x:label2id.get(x))
    data['content'] = data['content'].apply(lambda x:re.sub('[0-9][0-9]{5,}|[\\n]|[/\\r]|[【]|[】]|[[]|[]]|[A-Za-z0-9_]{5,}|[,]','',x))
    data['content'] = data['content'].apply(lambda x:re.sub('([0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]{1}|[0-9]{1}[1-9][0-9]{2}|[1-9][0-9]{3})-(((0[13578]|1[02])-(0[1-9]|[12][0-9]|3[01]))|((0[469]|11)-(0[1-9]|[12][0-9]|30))|(02-(0[1-9]|[1][0-9]|2[0-8])))|[0-9]{4}[年][0-9]{1,}[月][0-9]{1,}[日]|[0-9]{1,}[:0-9]{1,}[:0-9]{1,}','',x))
    # bert allow max_len == 512 包括[CLS][SEP]
    data['content'] = data['content'].apply(lambda x:x[:127]+x[-381:])
    
    data_arr = np.asarray(data)
    for i in range(len(data)):
        for corpus in data2:
            if corpus in data_arr[i][1]:
                data_arr[i][0] = 2
                continue
        for corpus in data1:
            if corpus in data_arr[i][1]:
                data_arr[i][0] = 1
                continue
        for corpus in data0:
            if corpus in data_arr[i][1]:
                data_arr[i][0] = 0
                continue
    data_new = pd.DataFrame(data_arr,columns=['label','columns'])
    # 取5000/19000
    # data_0_5000 = np.asarray(data_new[data_new['label']==0])[:5000]
    # data_new = data_new[data_new['label']!=0]
    # data_new = list(np.asarray(data_new[data_new['label']!=0]))
    # for i in data_0_5000:
    #     data_new.append(i)
    data_new = np.asarray(data_new)

    return data_new


def __call__():
    data_path = os.path.join(os.path.dirname(__file__),'data','data.csv')

    train_path = os.path.join(os.path.dirname(__file__),'data','train.pkl')
    valid_path = os.path.join(os.path.dirname(__file__),'data','valid.pkl')
    
    df_data = pd.read_csv(data_path)[['情绪等级标注结果','受理内容']]
    data = pre_process(df_data)

    _,_ = train_valid_split(data,train_path,valid_path)


def text2label(x):
    return text2label.get(x)
    
def seq_padding(token_list,padding=0):
    """
    填补每个batch的token到最大长度，0填充
    """
    L = max([len(token) for token in token_list])
    return np.array([np.concatenate([token,[padding]*(L-len(token))]) if len(token)<L else token for token in token_list])

def read_corpus(file_path):
    try:
        data = pickle.load(open(file_path,'rb+'))
        return data
    except EOFError:
        return 'read_corpus ERROR'

class DataGenerator():
    def __init__(self,data,tokenizer,batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = int(math.ceil(len(self.data)/self.batch_size))
        self.tokenizer = tokenizer
    
    def __len__(self):
        return self.steps
    
    def __iter__(self):
        batch_token_ids,batch_segment_ids,batch_labels = [],[],[]
        while True:
            data = self.data
            idxs = list(range(len(data)))
            np.random.shuffle(idxs)
            for index in idxs:
                # [[tokens],[tokens]]
                token_ids = [self.tokenizer._token_dict[TOKEN_CLS]]
                # 提取语料 逐字符编码
                for word in ''.join(self.data[index][1].strip().split(' ')):
                    w_token_ids = self.tokenizer.encode(word)[0][1:-1]
                    token_ids += w_token_ids

                token_ids += [self.tokenizer._token_dict[TOKEN_SEP]]
                segment_ids = [0] * len(token_ids)
                label = [data[index][0]]
                
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(label)
                if len(batch_token_ids)==self.batch_size:
                    batch_token_ids = seq_padding(batch_token_ids)
                    batch_segment_ids = seq_padding(batch_segment_ids)
                    yield [batch_token_ids,batch_segment_ids],to_categorical(batch_labels,num_classes=5)
                    batch_token_ids,batch_segment_ids,batch_labels = [],[],[]

    def forfit(self):
        while True:
            for d in self.__iter__():
                yield d

if __name__=='__main__':
    # 加载pretrained模型
    model_path = get_pretrained(PretrainedList.chinese_base)
    path = get_checkpoint_paths(model_path)
    # 加载token字典
    token_dict = load_vocabulary(path.vocab)
    tokenizer = Tokenizer(token_dict)

    train_path = os.path.join(os.path.dirname(__file__),'data','train.pkl')
    valid_path = os.path.join(os.path.dirname(__file__),'data','valid.pkl')
    if not os.path.exists(train_path):  __call__()

    train_data = read_corpus(train_path)
    valid_data = read_corpus(valid_path)

    trian_gen = DataGenerator(train_data,tokenizer).forfit()
    print(next(trian_gen))
    valid_gen = DataGenerator(valid_data,tokenizer).forfit()
    print(next(valid_gen))