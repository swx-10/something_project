import pickle,os,re,json,math
import numpy as np
import codecs
import unicodedata as ucd
from keras.utils.np_utils import to_categorical
from keras_bert import load_vocabulary,load_trained_model_from_checkpoint,Tokenizer,get_checkpoint_paths,gen_batch_inputs, tokenizer
from keras_bert.bert import TOKEN_CLS,TOKEN_SEP
from keras_bert.datasets import get_pretrained,PretrainedList

# 实体标签类别索引
label2id = {
    "O":0,
    "demandant": 1, 
    "legal_regulation": 2, 
    "dem_agent": 3, 
    "time": 4, 
    "cause": 5,
    "doc_type": 6,
    "judges": 7,
    "def_agent": 8,
    "defendant": 9,
    }
# 索引转标签实体
id2label = {str(v):k for k,v in label2id.items()}
def get_filename(path,filetype):# 输入路径/文件类型
    name = []
    for s in os.listdir(path):
        name.append(path +'/'+ s)
    # 输出由有后缀名的文件名组成的列表
    return name

# def split_file(file_list,split_ratio=0.9):
#     '''
#     按照split_ratio的比例切割数据集
#     :param file_list保存所有数据文档的list
#     :param split_ratio切割比例,默认为0.9
#     '''
#     file_nums = len(file_list)
#     train_num = int(file_nums*split_ratio)
#     train_file_list = file_list[:train_num]
#     test_file_list = file_list[train_num:]
#     return train_file_list,test_file_list


def file_preprocess(file_class,file_list):
    """
    传入标注文件路径的list,提取并保存到单个文件
    """
    corpus_path = './corpus/'+file_class+'.txt'
    with open(corpus_path,'w',encoding='utf-8') as f:
        for path in file_list:
            with open(path,'r',encoding='utf-8') as fh:
                data = fh.read()
            fh.close()
            f.write(data)
    f.close()
    return corpus_path

def corpus_process(corpus_path):
    """
    语料预处理，清洗
    """
    lines = []
    line = []
    with open(corpus_path,'r',encoding='utf-8',errors='ignore') as f:
        line = f.readlines()
        # "委 B-dem_ agent"|" O"|"  M-doc_type\n"
        for item in line:
            space_num = 0
            idx,idxs=[],[]
            if (item != '  O' and item[0]!=' ' and item != ''):
                for i in range(len(item)-1):
                    idxs.append(i)
                    if item[i]==' ':
                        space_num += 1
                        idx.append(i)
                if space_num > 1:
                    for i in idx[1:]:
                        idxs.remove(i)
                    item = "".join([item[int(i)] for i in idxs])+'\n'
                lines.append(item)
    with open(corpus_path,'w') as f:
        f.writelines(lines)
        f.close()

def label_process(corpus,label2id):
    start = np.zeros(len(corpus['text']),dtype='int')
    end = np.zeros(len(corpus['text']),dtype='int')
    for lb_index in corpus['labels'].items():
        if len(lb_index[1]) != 0:
            start[lb_index[1][0][0]] = label2id.get(lb_index[0])
            end[lb_index[1][0][1]] = label2id.get(lb_index[0])
    return start,end

def read_corpus(corpus_path):

    corpus_data = []
    with open(corpus_path,'r',encoding='utf-8') as f:
        content = json.load(f)
        for corpus in content:
            txt = [i for i in corpus.get('text')]
            start,end = label_process(corpus,label2id)
            corpus_data.append([txt,start,end])
    return corpus_data
'''
{   "id": 6, 
    "text": "。法定代表人：唐明，该公司董事长。", 
    "labels": { "cause": [], 
                "demandant": [], 
                "dem_agent": [[1, 9]], 
                "doc_type": [], 
                "defendant": [], 
                "def_agent": [], 
                "judges": [], 
                "legal_regulation": [], 
                "time": []}
}
'''

def read_json_corpus(corpus_path):
    """读取json格式的语料"""
    with codecs.open(corpus_path,'r',encoding='utf-8') as f:
        json_corpus = json.load(f)
    return json_corpus

def read_tags(tags_file):
    """读取标记文件"""
    with codecs.open(tags_file, 'r', encoding='utf-8') as f:
        tags = json.load(f)
    return tags

def seq_padding(X, padding=0):
    """
    对句向量X中所有token进行长度统计，取最长值。
    之后使用padding(0)补足到最长值
    param: X: 语料list
    param: padding: 填充值
    """
    # 每个句子的token长度
    L = [len(x) for x in X]
    # 最大句子token长度
    ML = max(L)
    return np.array([ np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])

class DataGenerator():
    """
    训练和测试用数据的生成器
    """
    def __init__(self,data,tokenizer,batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.steps = int(math.ceil(len(self.data)/self.batch_size))
        self.tokenizer = tokenizer
    
    def __len__(self):
        return self.steps
    
    def __iter__(self):
        batch_token_ids,batch_segment_ids,batch_labels_start,batch_labels_end = [],[],[],[]

        while True:
            # 随机提取语料
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            for i in idxs:
                if len(self.data[i][0]) > 500 or len(set(self.data[i][1]))==1 or len(set(self.data[i][2])) == 1:
                    continue
                # bert模型训练需要的token索引和label index
                token_ids = [self.tokenizer._token_dict[TOKEN_CLS]]

                # 提取语料和标签
                # 筛选正样本语料提取
                for w in self.data[i][0]:
                    # 对语料进行逐字符编码，用于后面bert训练
                    w_token_ids = self.tokenizer.encode(w)[0][1:-1]
                    token_ids += w_token_ids

                token_ids += [self.tokenizer._token_dict[TOKEN_SEP]]
                segment_ids = [0] * len(token_ids)
                
                start = [0]
                for j in self.data[i][1]:
                    start.append(j)
                start.append(0)

                end = [0]
                for j in self.data[i][2]:
                    end.append(j)
                end.append(0)
                
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels_start.append(start)
                batch_labels_end.append(end)

                if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                    batch_token_ids = seq_padding(batch_token_ids)
                    batch_segment_ids = seq_padding(batch_segment_ids)
                    batch_labels_start = seq_padding(batch_labels_start)
                    batch_labels_end = seq_padding(batch_labels_end)
                    # batch_labels_start (batch_size, ML, 11)
                    yield [batch_token_ids, batch_segment_ids],\
                           [to_categorical(batch_labels_start,num_classes=11),to_categorical(batch_labels_end,num_classes=11)]
                    
                    batch_token_ids,batch_segment_ids,batch_labels_start,batch_labels_end = [],[],[],[]


    def forfit(self):
        while True:
            for d in self.__iter__():
                yield d


if __name__ == "__main__":
    # # 版本迭代丢弃！！！！
    # # 初始标注文件目录
    # data_path = os.path.join(os.path.dirname(__file__),'corpus_json','BME')
    # file_list = get_filename(data_path,'.anns')
    # # # 按9:1的比例切割训练集验证集测试集
    # train_path,valid_path = split_file(file_list)
    # train_path = file_preprocess('train_corpus',train_path)
    # valid_path = file_preprocess('valid_corpus',valid_path)
    # # 文本清洗
    # corpus_process(train_path)
    # corpus_process(valid_path)

    # 加载中文预训练模型(缓存在当前用户.keras/datasets目录中)
    model_path = get_pretrained(PretrainedList.chinese_base)
    # 模型所在目录的path
    paths = get_checkpoint_paths(model_path)
    
    # 加载token字典
    token_dict = load_vocabulary(paths.vocab)
    # 创建tokenizer
    tokenizer = Tokenizer(token_dict)
    train_path = os.path.join(os.path.dirname(__file__),'corpus_json','BME_json','BME_train.json')
    valid_path = os.path.join(os.path.dirname(__file__),'corpus_json','BME_json','BME_test.json')

    train_data = read_corpus(train_path)
    valid_data = read_corpus(valid_path)

    train_gen = DataGenerator(train_data,tokenizer).forfit()
    next(train_gen)
    valid_gen = DataGenerator(valid_data,tokenizer).forfit()
    next(valid_gen)



