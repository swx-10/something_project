from keras_bert import Tokenizer
from werkzeug.datastructures import Headers
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
from run import load_bert,reverse_model
import unicodedata as ucd
import json,os,codecs,re
import numpy as np
from jsonrpc import JSONRPCResponseManager, dispatcher
from preprocess import read_json_corpus,DataGenerator,corpus_process,id2label


manager = JSONRPCResponseManager()

test_path = os.path.join(os.path.dirname(__file__),'corpus_json','test')
save_path = os.path.join(os.path.dirname(__file__),'spannet_NER.h5')
tokenizer = load_bert(just_load_token=True)
model = reverse_model(save_path)
id2label = {int(i):k for i,k in id2label.items()}

def read_txt_corpus(data_path):
    '''
    读取txt文件
    '''
    corpus_data,corpus = [],[]
    with open(data_path,'r',encoding='utf-8') as corpus_file:
        content = ucd.normalize('NFKC',corpus_file.read()).replace(' ', ' ') 
        corpus_tmp = re.split('\n\n|;|”|。',content)
        corpus = list(filter(lambda x: x!=('  O\n'and' O'and'\n'and''and'\n  O'and' O\n” O'),corpus_tmp))
        for corpu in corpus:
            if corpu!='' and corpu!=' O':
                tokens=[]
                for line in corpu.split('\n'):
                    if not(line=='' or line[0]==' 'or line[0]=='\t'or line=='”'):
                        # 提取每行中的token 和label
                        line = ''.join(line)
                        token = line.strip().split()
                        tokens.append(token)
                corpus_data.append([''.join(i) for i in tokens])
    return [i for j in range(len(corpus_data)) for i in corpus_data[j]] 

def predict_str(content,model,tokenizer):
    '''
    - content:data参数是直接传入字符串时推理结果
    '''
    ten_ids = np.asarray(tokenizer.encode(content)[0])
    seg_ids = np.asarray([0]*len(ten_ids))
    inputs = [ten_ids[None,:],seg_ids[None,:]]
    start,end = model.predict(inputs)
    # 概率
    start_seq = np.argmax(start,axis=-1).squeeze()
    end_seq = np.argmax(end,axis=-1).squeeze()
    # 去除首尾的[CLS][SEP]
    start = list(start_seq[1:-1])
    for i,st in enumerate(start):
        start[i] = id2label.get(st)
    end = list(end_seq[1:-1])
    for i,ed in enumerate(end):
        end[i] = id2label.get(ed)

    entity = None
    for i in range(len(start)):
        if start[i] != 'O':
            entity = start[i]
            idx_st = i
    for j in range(len(end)):
        if end[j] != 'O': 
            idx_ed = j
            break
        else: idx_ed=len(end)
    if entity:
        pre_data = {"content":content,"entity":content[idx_st:idx_ed]}
        return pre_data
    else:
        return None

def predict_file(data_path,model,tokenizer):
    '''
    传入的是txt文件路径时
    '''
    # corpus_process(data_path)
    corpus_list = read_txt_corpus(data_path)
    idx = np.random.randint(len(corpus_list))
    sent = corpus_list[idx]
    return predict_str(sent,model,tokenizer)


def predict(data_path):
    '''
    - data_path:判断是否是文件路径，如果不是，视为一段字符串直接进行NER
    '''
    if not (data_path):
        return {'status':1,'error message':"您未传入数据"}
    elif not os.path.exists(data_path):
        label = predict_str(data_path,model,tokenizer)
        return label
    else:
        label = predict_file(data_path,model,tokenizer)
        return label


@Request.application
def application(request):
    # 手工注册rpc服务
    dispatcher['predict'] = predict
    response = manager.handle(request.get_data(cache=False, as_text=True), dispatcher)
    return Response(response.json, mimetype='application/json',headers= Headers([('Access-Control-Allow-Origin', '*')]))

if __name__ == '__main__':
    # run_simple('localhost', 8090, application)
    path = os.path.join(os.path.dirname(__file__),'corpus_json','test','test.txt')
    while True:
        choice = input('是否开始/继续(y/n)').lower()
        a = predict(path)
        print(a)
        if choice == 'n':
            break