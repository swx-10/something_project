import os,json,math
from keras_bert import tokenizer
from keras_bert.datasets.pretrained import PretrainedList, get_pretrained
from keras_bert.util import get_checkpoint_paths
import numpy as np
from random import choice

from preprocess import split_dataset
from tokenizer import OurTokenizer,read_token


maxlen = 160

class DataGenerator:
    def __init__(self,data,tokenizer,predicate2id,batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = int(math.ceil(len(self.data)/self.batch_size))
        self.tokenizer = tokenizer
        self.predicate2id = predicate2id
        self.num_classes = len(predicate2id)

    def __len__(self):
        return self.steps

    def forfit(self):
        while True:
            for d in self.__iter__():
                yield d 
                
    def __iter__(self):
        def seq_padding(X,padding=0):
                '''最大长度字段填充X中所有序列（内嵌方法）'''
                L = [len(x) for x in X]
                ML = max(L)
                return np.array([
                    np.concatenate([x,[padding] * (ML - len(x))]) if len(x) < ML else x for x in X
                    ])
        def find_span_from_list1(list1,list2):
            '''寻找原串中子串的始末位置，原串中不存在则返回-1'''
            n_list2 = len(list2)
            for i in range(len(list1)):
                if list1[i:i+n_list2] == list2: return i
            return -1
            
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            T1,T2,S1,S2,K1,K2,O1,O2 = [],[],[],[],[],[],[],[]
            for i in idxs:
                d = self.data[i]
                text = d['text'][:maxlen]
                tokens = self.tokenizer.tokenize(text)
                # 目标items {(主体st,主体ed):(客体st,客体ed,关系id)}  
                # {(st,ed) :(st,ed,predicate_index)}
                items = {}
                for sp in d['spo_list']:
                    sp = (self.tokenizer.tokenize(sp[0])[1:-1],sp[1],self.tokenizer.tokenize(sp[2])[1:-1])
                    subjectid = find_span_from_list1(tokens,sp[0])
                    objectid = find_span_from_list1(tokens,sp[2])
                    if subjectid != -1 and objectid != -1:
                        # key存储subjectid_token的[start,end]
          
                        key = (subjectid,subjectid+len(sp[0]))
                        if key not in items:    items[key] = []
                        items[key].append(( objectid,
                                            objectid + len(sp[2]),
                                            self.predicate2id[sp[1]]))
                if items:
                    # tokenizer对text编码后作为bert输入，T1(token_id) T2(seg_id)
                    t1,t2 = self.tokenizer.encode(first=text)
                    # bert的输入 [T1,T2]
                    # T1为text的[CLS,...,SEP]
                    # T2为长度为text的seg_list
                    T1.append(t1)
                    T2.append(t2)

                    # 存储subject的start、end占位符
                    # subject的span 两组,一组start,一组end
                    s1,s2 = np.zeros(len(tokens)),np.zeros(len(tokens))
                    for j in items.keys():
                        s1[j[0]] = 1
                        s2[j[1] - 1] = 1

                    # 提取当前所有预料中subject的start_index,end_index
                    k1,k2 = np.array(list(items.keys())).T
                    # 随机选择subject索引
                    k1 = choice(k1)
                    k2 = choice(k2[k2>=k1])
                    # 根据text长度创建两个全0矩阵[token_len,predicate_class]
                    # 存贮object的start_index,end_index
                    o1,o2 = np.zeros((len(tokens),self.num_classes)),np.zeros((len(tokens),self.num_classes))
                    # 随机采样一组subject索引，
                    # 标记o1,o2中对应object位置的第predcate个 mask为1（标记为predicate）
                    for j in items.get((k1,k2),[]):
                        o1[j[0]][j[2]] = 1
                        o2[j[1]-1][j[2]] = 1
                    S1.append(s1)       # subject在text中的start_mask
                    S2.append(s2)       # subject在text中的end_mask
                    K1.append([k1])     # subject在text中的[start]
                    K2.append([k2-1])   # subject在text中的[end]
                    O1.append(o1)       # object在start位置的predicate
                    O2.append(o2)       # object在end位置的predicate

                    if len(T1) == self.batch_size or i == idxs[-1]:
                        T1 = seq_padding(T1)    # text token
                        T2 = seq_padding(T2)    # seg token
                        S1 = seq_padding(S1)    # sub start mask
                        S2 = seq_padding(S2)    # sub end mask
                        O1 = seq_padding(O1,np.zeros(self.num_classes))     # obj predicate_start
                        O2 = seq_padding(O2,np.zeros(self.num_classes))     # obj predicate_end
                        K1,K2 = np.array(K1),np.array(K2)   # subject start end
                        yield [T1, T2, S1, S2, K1, K2, O1, O2], None
                        T1, T2, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], [], []

if __name__ == "__main__":
    bert_path = get_pretrained(PretrainedList.chinese_base)
    paths = get_checkpoint_paths(bert_path)

    token_dict = read_token(paths.vocab)
    tokenizer = OurTokenizer(token_dict)
    # 加载语料集
    train_file = os.path.join(os.path.dirname(__file__),'data','train_data.json')
    dev_file = os.path.join(os.path.dirname(__file__),'data','dev_data.json')
    schema_file = os.path.join(os.path.dirname(__file__),'data','schema.json')
    train_data = json.load(open(train_file))
    dev_data = json.load(open(dev_file))
    _,predicate2id = json.load(open(schema_file))

    # 完整数据集
    total_data = []
    total_data.extend(train_data)
    total_data.extend(dev_data)

    train_data,test_data = split_dataset(total_data)
    gen = DataGenerator(train_data,tokenizer,predicate2id).__iter__()
    print(next(gen))