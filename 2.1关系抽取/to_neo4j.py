from neo4j import GraphDatabase
import json,os
from tqdm import tqdm

uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "qwe123"))

def get_data(data_path):
    with open(data_path,encoding='utf-8') as f:
        data = [''.join(i.split('\n')) for i in f.readlines()]
    f.close()
    spo_list = []
    for text in data:
        text = json.loads(text).get('spo_list')
        for _spo in text:
            spo_list.append(_spo)
    return spo_list

def get_schema(schemas_path):
    with open(schemas_path,'r',encoding='utf-8') as f:
        schemas = [''.join(i.split('\n')) for i in f.readlines()]
    f.close()
    schema_list = []
    for i in schemas:
        i = json.loads(i)
        schema_list.append(i)
    schema_dict = {p.get('predicate'):[p.get('subject_type'),p.get('object_type')] for p in schema_list}
    return schema_dict

def create_spo_node(tx, subject, objects, predicate, subject_type=None, object_type=None):
    cql = f"merge (sub:{subject_type} {{subject:'{subject}'}}) merge (sub)-[:{predicate}]->(:{object_type} {{object: '{objects}'}})"
    tx.run(cql)

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__),'test_pred.json')
    schemas_path = os.path.join(os.path.dirname(__file__),'data_ori','all_50_schemas.json')
    spo_list = get_data(data_path)
    schema_dict = get_schema(schemas_path)

    with driver.session() as session:
        for item in tqdm(spo_list):
            subject = item.get('subject')
            objects = item.get('object')
            predicate = item.get('predicate')
            subject_type,object_type = schema_dict.get(predicate)
            session.write_transaction(create_spo_node, subject, objects, predicate, subject_type, object_type)
            subject,objects,predicate,subject_type,object_type = [],[],[],[],[]
    driver.close()