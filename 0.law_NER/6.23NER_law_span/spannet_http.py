from flask import Flask, request
from wsgiref.simple_server import make_server
import json,os
import numpy as np
# from tf1_mnist import generate_model, reload_session, predict
from run import load_bert,reverse_model
from spannet_jsonrpc import predict

app = Flask(__name__) 
# train, output, x, y = generate_model()
# sess = reload_session()

test_path = os.path.join(os.path.dirname(__file__),'corpus_json','test')
save_path = os.path.join(os.path.dirname(__file__),'spannet_NER.h5')
tokenizer = load_bert(just_load_token=True)
model = reverse_model(save_path)

@app.route('/pred', methods=['POST'])
def post_process():
    data_path = request.form['data']
    print(data_path)
    # 接收的str数据转换为numpy数组
    # im = np.array(json.loads(data))
    label = predict(data_path)
    return label


if __name__ == '__main__':
    # 正式服务
    print('Flask服务启动')
    # server = make_server('127.0.0.1', 8089, app)
    # server.serve_forever()
    app.run(host='127.0.0.1', port=8089)