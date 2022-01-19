import json
import os
from bottle import get, run, request, response, static_file
from py2neo import Graph
from py2neo.data import Node, Relationship
from py2neo.matching import NodeMatcher
from tqdm import tqdm

graph = Graph(password='147258')
# 查询节点是否存在对象
matcher = NodeMatcher(graph)
# 关系数据json文件路径
jsonData_path = os.path.join(os.path.dirname(__file__),'test_pred.json')
# 关系对应实体类别
schemas_path = os.path.join(os.path.dirname(__file__), 'data_ori', 'all_50_schemas.json')

def get_data(data_path):
    with open(data_path, encoding='utf-8') as f:
        data = f.read().splitlines()
    result = []
    for i in data:
        # result.append(json.loads(json_obj))   
        if json.loads(i).get('spo_list'):
            result += json.loads(i).get('spo_list')
    return result

def get_relation(schemas_path):
    with open(schemas_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    result = {}
    for d in data:
        type_dict = json.loads(d.strip())
        result[type_dict['predicate']] = (type_dict['object_type'], type_dict['subject_type'])
    return result


@get('/create')
def get_index():
    data = get_data(jsonData_path)
    label = get_relation(schemas_path)

    for i in tqdm(data):
        if not label[i[0]['predicate']]:
            continue
        o_type, s_type = label[i['predicate']]
        # 查找出发节点是否存在
        o1 = list(matcher.match(s_type, name=i['subject']))
        # 查找结束节点是否存在
        o2 = list(matcher.match(o_type, name=i['object']))
        # 如果出发节点(subject)存在, 结束节点(object)不存在
        if len(o1) and not len(o2):

            end_node = Node(o_type, name=i['object'])
            graph.create(Relationship(o1[0], i['predicate'], end_node))
        # 如果出发节点(subject)不存在, 结束节点(object)存在
        elif not len(o1) and len(o2):
            start_node = Node(s_type, name=i['subject'])
            graph.create(Relationship(start_node, i['predicate'], o2[0]))
        # 都存在
        elif len(o1) and len(o2):
            graph.create(Relationship(o1[0], i['predicate'], o2[0]))
        # 都不存在
        else:
            start_node = Node(s_type, name=i['subject'])
            end_node = Node(o_type, name=i['object'])
            graph.create(Relationship(start_node, i['predicate'], end_node))

    print('关系创建完成！！')
    # return '关系创建完成'


if __name__ == '__main__':
    run(port=8090)