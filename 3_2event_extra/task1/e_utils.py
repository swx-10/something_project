import numpy as np
import logging as log
import re,json
from generator import id2label

max_entity_len = 6
id2label = {int(k):v for k,v in id2label.items()}


def predict_str(content,model,tokenizer,f=None,model_pattern='strict'):
    '''
    - model_pattern:'tolerant' or 'strict'构建entity_idx_list方法不同
    '''
    ten_ids = [101]
    for alpha in content:
        ten_ids += tokenizer.encode(alpha)[0][1:-1]
    ten_ids = np.asarray(ten_ids+[102])
    seg_ids = np.asarray([0]*len(ten_ids))
    inputs = [ten_ids[None,:],seg_ids[None,:]]
    start,end = model.predict(inputs)
    start_seq = list(np.argmax(start,axis=-1).squeeze()[1:-1])
    end_seq = list(np.argmax(end,axis=-1).squeeze()[1:-1])
    # pre_data = {'text':content}
    attribute = []
    if len(set(start_seq)) == 1 and len(set(end_seq))==1:
        return None
    else:
        # label
        st_label = [id2label.get(st) for st in start_seq]
        ed_label = [id2label.get(st) for st in end_seq]
        attribute = []
        # index
        st_idx = np.argwhere(np.asarray(st_label) != 'O')
        ed_idx = np.argwhere(np.asarray(ed_label) != 'O')
        if len(st_idx) == 1:
            st_idx = [int(i) for i in st_idx]
        else:
            st_idx = [int(i) for i in st_idx.squeeze()]

        if len(ed_idx) == 1:
            ed_idx = [int(i) for i in ed_idx]
        else:
            ed_idx = [int(i) for i in ed_idx.squeeze()]
        # 穷举构建一个所有可能存在的实体集合
        entity_idx_list = []
        # 全匹配模式
        if model_pattern == 'strict':
            if len(st_idx)==len(ed_idx) and len(st_idx)!=0 and len(ed_idx)!=0:
                entity_idx_list = [(st_idx[i],ed_idx[i]) for i in range(len(st_idx))]
            else:
                return None
        # 宽容匹配模式
        elif model_pattern == 'tolerant':
            used_idxs = []
            for st in st_idx:
                for ed in ed_idx:
                    if abs(ed - st) < 10 and ed > st and st not in used_idxs and ed not in used_idxs:
                        entity_idx_list.append((st,ed))
                        used_idxs.append(st)
                        used_idxs.append(ed)
                        continue

            # 排除使用过的index,添加猜测可能存在的实体span
            used_idxs = set(np.asarray(entity_idx_list).reshape(1,-1).squeeze())
            for st in st_idx:
                if st not in used_idxs:
                    if (st + max_entity_len) < len(content): entity_idx_list.append((st,st+1))
                    else:entity_idx_list.append((st,len(content)))
            for ed in ed_idx:
                if ed not in used_idxs:
                    entity_idx_list.append((ed-1,ed))

        for i in range(len(entity_idx_list)):
            span = entity_idx_list[i]
            p = span[0]
            q = span[1]
            a = st_label[p]
            b = ed_label[q]
            type = a if a != 'O' else b
            attribute.append({  'type':type,
                                'entity':content[p:q+1],
                                'start':p,
                                'end':q})
        if f:   
            s = json.dumps(attribute,ensure_ascii=False)
            f.write(s+'\n')
        # pre_data = {"content":content,"attribute":attribute}        
        return attribute

def pre_dict(test_data,f,model_pattern,pattern,model,tokenizer):
    batch = int(input('请输入要预测的样本数：'))
    while True:
        for i in range(batch):
            idx = np.random.randint(len(test_data))
            content = test_data[idx].get('text')
            if len(content) > 510:  content = content[:510]
            content = ''.join(content.split())
            content = re.sub(pattern,'',content)
            attr = predict_str(content,model,tokenizer,f,model_pattern)
            log.info(attr)

        choice = input('是否继续(y/n)').lower()
        if choice == 'n':
            f.close()
            break
        else:
            batch = int(input('请输入要预测的样本数：'))