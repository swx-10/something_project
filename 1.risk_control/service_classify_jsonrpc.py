from typing_extensions import runtime
from keras_bert import get_custom_objects
from werkzeug.datastructures import Headers
from werkzeug.wrappers import  Request,Response, response
from werkzeug.serving import run_simple
from jsonrpc import JSONRPCResponseManager,dispatcher, manager
from run import load_bert,predict,model_name,reverse_model
from keras.models import load_model
import numpy as np
import os,re

manager = JSONRPCResponseManager()

tokenizer = load_bert(just_load_token=True)
# model_path = os.path.join(os.path.dirname(__file__),'classify_5_5000.h5')
model_path = os.path.join(os.path.dirname(__file__),model_name)
model = reverse_model(model_path)

def dec(func):
    def corpus_process(*args):
        sents = []
        for sent in args:
            sent = re.sub('[0-9][0-9]{6,}|[\\n]|[/\\r]|[【]|[】]|[[]|[]]|[A-Za-z0-9_]{5,}|[,]','',sent)
            sent = re.sub('([0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]{1}|[0-9]{1}[1-9][0-9]{2}|[1-9][0-9]{3})-(((0[13578]|1[02])-(0[1-9]|[12][0-9]|3[01]))|((0[469]|11)-(0[1-9]|[12][0-9]|30))|(02-(0[1-9]|[1][0-9]|2[0-8])))|[0-9]{4}[年][0-9]{1,}[月][0-9]{1,}[日]|[0-9]{1,}[:0-9]{1,}[:0-9]{1,}','',sent)
            if len(sent)>510:sent = sent[:127] + sent[-381:]
            sents.append(sent)
        output = func(sents)
        return output
    return corpus_process

@dec
def read(sent):
    label = []
    label = predict(sent,model,tokenizer)
    return {'label':label}

@Request.application
def application(request):
    dispatcher['read'] = read
    response = manager.handle(request.get_data(cache=False,as_text=True),dispatcher)
    return Response(response.json,mimetype='application/json',headers=Headers([('Access-Control-Allow-Origin','*')]))

if __name__=="__main__":
    run_simple('127.0.0.1',8888,application)
    # read(['用户来电表示其宽月的费用来算，已向用户解释，用户对已意见非常大，请协助处理，谢谢！",邵桂文2113已经联系用户处理，请贵司回单谢谢',
    #         '用户来电表示其宽何现在又欠费停机了，前台经核实，用户号码因手机号码18046909786是预开通状态，所以宽带按照150元/月的费用来算，已向用户解释，用户对已意见非常大，请协助处理，谢谢！",邵桂文2113已经联系用户处理，请贵司回单谢谢,用户来电表示其宽月的费用来算，已向用户解释，用户对已意见非常大，请协助处理，谢谢！",邵桂文2113已经联系用户处理，请贵司回单谢谢',
    #         '经核实，用户号码因手机号码18046909786是预开通状态，所以宽带按照150元/月的费用'])