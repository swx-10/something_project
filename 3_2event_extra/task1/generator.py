import os,json,math,re,logging
import tensorflow.keras.backend as K
from keras_bert import Tokenizer
from keras_bert.bert import TOKEN_CLS, TOKEN_MASK, TOKEN_SEP
from keras_bert.datasets.pretrained import PretrainedList, get_pretrained
from keras_bert.loader import load_vocabulary
from keras_bert.util import get_checkpoint_paths
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
logging.basicConfig(level=logging.INFO)

data_path = os.path.join(os.path.dirname(__file__),'data','task1_train.txt')
elements_file = os.path.join(os.path.dirname(__file__),'data','elements.txt')
train_path = os.path.join(os.path.dirname(__file__),'data','train.txt')
valid_path = os.path.join(os.path.dirname(__file__),'data','valid.txt')
# with open(elements_file,'r',encoding='utf-8') as f:
#     label2id = {key:i+1  for i,key in enumerate(json.load(f))}
# id2label = {str(v):k for k,v in label2id.items()}
label2id = {'O':0,
            '受害人': 1, '嫌疑人': 2, '资损金额': 3, 
            '案发时间': 4, '支付渠道': 5, '涉案平台': 6, 
            '受害人身份': 7, '交易号': 8, '订单号': 9, 
            '手机号': 10, '银行卡号': 11, '身份证号': 12,
            '案发城市': 13}

id2label = {'0':'O',
            '1': '受害人', '2': '嫌疑人', '3': '资损金额',
            '4': '案发时间', '5': '支付渠道', 
            '6': '涉案平台', '7': '受害人身份', '8': '交易号', 
            '9': '订单号', '10': '手机号', '11': '银行卡号', 
            '12': '身份证号','13': '案发城市'}

def read_data(path):
    return json.load(open(path,'r',encoding='utf-8'))

def train_valid_split(data_path,train_path,valid_path,split_rate=0.9):
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    data_nums = len(data)
    np.random.shuffle(data)
    with open(train_path,'w',encoding='utf-8') as f:
        json.dump(data[:int(data_nums*split_rate)],f)
    f.close()
    with open(valid_path,'w',encoding='utf-8') as f:
        json.dump(data[int(data_nums*split_rate):],f)
    f.close()

class DataGenerator:
    def __init__(self,data,tokenizer,batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.steps = int(math.ceil(len(self.data)/self.batch_size))
        self.tokeinzer = tokenizer
    
    def __len__(self):
        return self.steps
    
    def forfit(self):
        while True:
            for d in self.__iter__():
                yield d

    def seq_padding(self,X,padding=0):
        '''最大长度字段填充X中所有序列（内嵌方法）'''
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([np.concatenate([x,[padding] * (ML - len(x))])  if len(x) < ML else x for x in X])
    
    def __iter__(self):
        batch_token_ids,batch_seg_ids,batch_label_start,batch_label_end,batch_length = [],[],[],[],[]
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            for i in idxs:
                token_ids = [self.tokeinzer._token_dict[TOKEN_CLS]]
                data = self.data[i]
                text = data.get('text')
                if len(text) > 510:
                    continue
                for w in text:
                    if w==' ':  token_ids += [self.tokeinzer._token_dict[TOKEN_MASK]]
                    else:
                        w_token_ids = self.tokeinzer.encode(w)[0][1:-1]
                        token_ids += w_token_ids
                token_ids += [self.tokeinzer._token_dict[TOKEN_SEP]]
                segment_ids = [0] * len(token_ids)

                start = [0] * len(token_ids)
                end = [0] * len(token_ids)
                length = [0] * len(token_ids)
                
                for attr in data['attributes']:
                    type_id = label2id.get(attr.get('type'))
                    start[attr.get('start')+1] = type_id
                    end[attr.get('start')+2] = type_id
                    length[attr.get('start')+1] = attr.get('end') - attr.get('start') + 1
                
                batch_token_ids.append(token_ids)
                batch_seg_ids.append(segment_ids)
                batch_length.append(length)
                                
                batch_label_start.append(start)
                batch_label_end.append(end)

                if len(batch_token_ids)==self.batch_size or i == idxs[-1]:
                    batch_token_ids = self.seq_padding(batch_token_ids)
                    batch_seg_ids = self.seq_padding(batch_seg_ids)
                    batch_length = self.seq_padding(batch_length)

                    batch_label_start = self.seq_padding(batch_label_start)
                    batch_label_end = self.seq_padding(batch_label_end)
                    
                    yield [batch_token_ids, batch_seg_ids],\
                        [to_categorical(batch_label_start,num_classes=14), 
                        to_categorical(batch_label_end,num_classes=14),
                        to_categorical(batch_length,num_classes=30)
                        # np.expand_dims(batch_length,2)
                        ]

                    batch_token_ids,batch_seg_ids,batch_label_start,batch_label_end,batch_length = [],[],[],[],[]

if __name__=='__main__':
    if not (os.path.exists(train_path) and os.path.exists(valid_path)):
        train_valid_split(data_path,train_path,valid_path)
    
    model_path = get_pretrained(PretrainedList.chinese_base)
    paths = get_checkpoint_paths(model_path)
    token_dict = load_vocabulary(paths.vocab)
    tokenizer = Tokenizer(token_dict)   #读取中文预训练模型 生成tokenizer

    train_data = read_data(train_path)
    valid_data = read_data(valid_path)
    train_gen = DataGenerator(train_data,tokenizer).forfit()
    logging.info(next(train_gen))
    valid_gen = DataGenerator(valid_data,tokenizer).forfit()
    next(valid_gen)

    



