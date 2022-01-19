import os
import json
import logging as log

log.basicConfig(level=log.NOTSET)

# 语料文件
event_train_file = os.path.join(os.path.dirname(__file__),'data','task2_train.txt')
# 类别文件
event_tag_file = os.path.join(os.path.dirname(__file__),'data','event_tags.json')

def fetch_types():
    """提取事件类型和结果类型"""
    result_types = set()
    reason_types = set()
    items = [json.loads(line) for line in open(event_train_file)]

    for item in items:
        for result in item['result']:
            reason_types.add(result['reason_type'])
            result_types.add(result['result_type'])
    
    return reason_types,result_types

def fetch_type_elements(reason_types, result_types):
    """提取事件和结果三要素(地域、产品、行业)"""
    items = [json.loads(line) for line in open(event_train_file)]

    reason_dict = { r:{'product':[],'region':[], 'industry':[] } for r in reason_types }
    result_dict = { r:{'product':[],'region':[], 'industry':[] } for r in result_types }

    for item in items:
        for result in item['result']:
            if result['reason_product'] != '' and result['reason_product'] not in reason_dict[result['reason_type']]['product']:
                reason_dict[result['reason_type']]['product'].append(result['reason_product'])    # 事件产品
            if result['reason_region'] != '' and result['reason_region'] not in reason_dict[result['reason_type']]['region']:
                reason_dict[result['reason_type']]['region'].append(result['reason_region'])     # 事件地域
            if result['reason_industry'] != '' and result['reason_industry'] not in reason_dict[result['reason_type']]['industry']:
                reason_dict[result['reason_type']]['industry'].append(result['reason_industry'])   # 事件行业
            if result['result_product'] != '' and result['result_product'] not in result_dict[result['result_type']]['product']:
                result_dict[result['result_type']]['product'].append(result['result_product'])    # 结果产品
            if result['result_region'] != '' and result['result_region'] not in result_dict[result['result_type']]['region']:
                result_dict[result['result_type']]['region'].append(result['result_region'])     # 结果地域
            if result['result_industry'] != '' and result['result_industry'] not in result_dict[result['result_type']]['industry']:
                result_dict[result['result_type']]['industry'].append(result['result_industry'])   # 结果行业

    
    return reason_dict, result_dict

if __name__ == '__main__':
    reason_types,result_types = fetch_types()

    log.info(result_types)
    log.info(reason_types)

    reasons, results = fetch_type_elements(reason_types,result_types)

    with open(event_tag_file,'w') as f:
        json.dump([reasons, results],f)
    
