import codecs
from keras_bert import Tokenizer

def read_token(bert_dict_path):
    '''读取bert的token_dict'''
    token_dict = {}
    with codecs.open(bert_dict_path,'r','utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict

# 自定义Tokenizer类，把空格作为[unused1]类
class OurTokenizer(Tokenizer):
    def _tokenize(self,text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R
        
