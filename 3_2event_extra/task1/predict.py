import re,os
import numpy as np
from genericpath import exists

from tensorflow.python.keras import backend
from generator import read_data
from entitpy_predict import pre_dict
from entitpy_predict import reload_model,model_path,predict_str,load_bert
from werkzeug.datastructures import Headers
from werkzeug.wrappers import Request,Response
from werkzeug.serving import run_simple
from jsonrpc import JSONRPCResponseManager,dispatcher,manager
import logging as log

log.basicConfig(level = log.NOTSET)

manager = JSONRPCResponseManager()

test_path = os.path.join(os.path.dirname(__file__),'data','task1_eval_data.txt')
test_result_path = os.path.join(os.path.dirname(__file__),'test_result.txt')

tokenizer = load_bert(just_load_token=True)
model = reload_model(model_path)
pattern = re.compile('[,.:;，。；：“”#、| !()@↓*〔〕《》]')
test_data = read_data(test_path)

if not exists(test_result_path):    f = open(test_result_path,'w',encoding='utf-8')
else:   f = open(test_result_path,'a',encoding='utf-8')

def predict():
    batch = 10
    result = []
    while True:
        idx = np.random.randint(len(test_data))
        content = test_data[idx].get('text')
        if len(content) > 510:  content = content[:510]
        content = ''.join(content.split())
        content = re.sub(pattern,'',content)
        attr = predict_str(content,model,tokenizer,f)
        log.info(attr)
        result.append(attr)
        if len(result) == batch:
            return result
    

@Request.application
def application(request):
    dispatcher['predict'] = predict
    response = manager.handle(request.get_data(cache=False,as_text=True),dispatcher)
    return Response(response.json,mimetype='application/json',headers=Headers([('Access-Control-Allow-Origin','*')]))


if __name__=="__main__":
    run_simple('0.0.0.0',8888,application)