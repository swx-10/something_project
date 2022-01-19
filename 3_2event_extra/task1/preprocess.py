import os,json,re
import logging as log
from tqdm import tqdm

log.basicConfig(level=log.NOTSET)

# 语料文件
train_path = os.path.join(os.path.dirname(__file__),'data','task1_train.txt')
eval_path = os.path.join(os.path.dirname(__file__),'data','task1_eval_data.txt')
# 类别文件输出
level_file = os.path.join(os.path.dirname(__file__),'data','levels.txt')
# 要素文件输出
elements_file = os.path.join(os.path.dirname(__file__),'data','elements.txt')


def fetch_levels():
    """提取语料中的事件类别"""
    items = [json.loads(line) for line in open(train_path,encoding='utf-8')]
    # 事件分级列表
    level1,level2,level3 = [],[],[]
    for item in items[0]:
        level1.append(item['level1'])
        level2.append(item['level2'])
        level3.append(item['level3'])
    # 不同类别的样本数量
    log.info('level1 length:%d'%len(level1))
    log.info('level2 length:%d'%len(level2))
    log.info('level3 length:%d'%len(level3))
    # 类别去重
    level1 = set(level1)
    level2 = set(level2)
    level3 = set(level3)

    return {'level1':list(level1), 
            'level2':list(level2),
            'level3':list(level3)}

def fetch_events():
    """提取语料中的事件要素和要素值"""
    items = [json.loads(line) for line in open(train_path,encoding='utf-8')]
    # 事件要素
    elements = dict()
    for item in items[0]:
        attrs = item['attributes']
        for atr in attrs:
            elements[atr['type']] = elements.get(atr['type'],[])
            elements[atr['type']].append(atr['entity'])
    return elements

def splittext(text,attributes):
    pattern = re.compile(r"[.|。|？|?|！|!| |、|,|，]") #匹配这些表示句子结束的标点符号
    match = re.finditer(pattern,text)#匹配到了
    match_index = [i.span()[1] for i in match]#找到匹配到的 位置，即索引值
    m = 510
    n = 0
    data,atts = [],[]
    for i in range(int(len(text)/510)+1): 
        att = []
        if m >= max(match_index):
            data.append(text[n:])
            for i in attributes:
                  if n <= i["end"]:
                    i["start"] = i["start"]-n
                    i["end"] = i["end"]-n
                    att.append(i)
            atts.append(att)
            break
        a = [j for j in match_index if j<=m]
        a = max(a)
        for i in attributes:
            if n <= i["end"] <= a:
                i["start"] = i["start"]-n
                i["end"] = i["end"]-n
                att.append(i)
        atts.append(att)
        data.append(text[n:a])
        n = a
        m = a + 510
    return data,atts
def preprocess(path):
    data_all = []
    with open(path,"rb") as f:
        lines = json.load(f)
        for line in tqdm(lines):
            data_new = {}
            data = line
            text = data["text"]
            att = data['attributes']
            if len(text) > 510:
                text,att = splittext(text,att)
                for i in range(len(text)):
                    data_new = {}
                    data_new['text_id'] = data['text_id']
                    data_new['text'] = text[i]
                    data_new['level1'] = data['level1']
                    data_new['level2'] = data['level2']
                    data_new['level3'] = data['level3']
                    data_new['attributes'] = att[i]
                    data_all.append(data_new)
                continue
            else:
                data_new['text_id'] = data['text_id']
                data_new['text'] = text
                data_new['level1'] = data['level1']
                data_new['level2'] = data['level2']
                data_new['level3'] = data['level3']
                data_new['attributes'] = att
            data_all.append(data_new)
    f.close()
    json.dump(data_all,open(path,'w',encoding='utf-8'))
    return print('long term preprocess success!!!')

def split(text):
    pattern = re.compile(r"[.|。|？|?|！|!| |、|,|，]")
    match = re.finditer(pattern,text)
    match_index = [i.span()[1] for i in match]
    m = 510
    n = 0
    data = []
    for i in range(int(len(text)/510)+1):
        if m >= max(match_index):
            data.append(text[n:])
            break
        a = [j for j in match_index if j<=m]
        a = max(a)
        data.append(text[n:a])
        n = a
        m = a + 510
    return data

def preprocess_eval_str(path):
    data_all = []
    with open(path,"rb") as f:
        lines = json.load(f)
        #lines = f.read().splitlines()
        for line in tqdm(lines):
            data = line
            text = data["text"]
            #att = data['attributes']
            data_new = {}
            # data = json.loads(line)
            #data = line
            #text = data[0]["text"]
            # att = data[0]['attributes']
            if len(text) > 510:
                text = split(text)
                for i in range(len(text)):
                    data_new = {}
                    data_new['text_id'] = data['text_id']
                    data_new['text'] = text[i]
                    data_new['level1'] = data['level1']
                    data_new['level2'] = data['level2']
                    data_new['level3'] = data['level3']
                    data_all.append(data_new)
                continue
            else:

                data_new['text_id'] = data['text_id']
                data_new['text'] = text
                data_new['level1'] = data['level1']
                data_new['level2'] = data['level2']
                data_new['level3'] = data['level3']
            data_all.append(data_new)
    f.close()
    json.dump(data_all,open(path,'w',encoding='utf-8'))
    return print('long term preprocess success!!!')

if __name__ == '__main__':
    levels = fetch_levels()
    with open(level_file,'w') as f:
        json.dump(levels, f)

    # 去重后类别数    
    log.info('level1 items:%d'%len(levels['level1']))
    log.info('level2 items:%d'%len(levels['level2']))
    log.info('level3 items:%d'%len(levels['level3']))
    # 类别内容
    log.info('level1 items:%r'%levels['level1'])
    log.info('level2 items:%r'%levels['level2'])
    log.info('level3 items:%r'%levels['level3'])

    elements = fetch_events()
    log.info('elements length:%d'%len(elements)) #所有的案件元素，就是标签 共13个
    for k,v in elements.items():
        log.info('%s: %r'%(k,len(v)))
        # log.info('%s: %r'%(k,v))

    with open(elements_file,'w') as f:
        json.dump(list(elements.keys()),f)

    # 长文本处理，切割成新的语料
    preprocess(train_path)
    preprocess_eval_str(eval_path)