import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import nltk
import re
from urllib.parse import unquote
from gensim.models.word2vec import Word2Vec


# 定义GeneSeg函数，用于处理输入的payload
def GeneSeg(payload):
    # 数字泛化为"0"
    payload = payload.lower()
    payload = unquote(unquote(payload))
    payload, num = re.subn(r"\d+", "0", payload)
    # 替换url为 http://u
    payload, num = re.subn(
        r"(http|https)://[a-zA-Z0-9\.@&/#!#\?]+", "http://u", payload
    )
    # 分词
    r = """
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    """
    return nltk.regexp_tokenize(payload, r)


# 导入模型
model = tf.keras.models.load_model("xss-models/trained_xssdec_model.h5")

# 导入tokenizer对象
with open("xss-models/tokenizer.pickle", "rb") as file:
    tokenizer = pickle.load(file)


# 导入词向量
def predict_single_sentence(sentence):
    # 对输入语句进行预处理
    processed_sentence = GeneSeg(sentence)  # 使用GeneSeg函数进行处理，确保与训练时的预处理一致
    processed_sentence = [
        word if word in tokenizer.word_index else "UNK" for word in processed_sentence
    ]  # 根据词汇表替换词汇

    # 将预处理后的语句转换为序列
    sequence = tokenizer.texts_to_sequences([processed_sentence])

    # 进行填充
    sequence = pad_sequences(sequence, maxlen=516)  # X是训练时的输入序列长度

    # 进行预测
    probability = model.predict(sequence)
    prediction = (probability > 0.5).astype(int)

    return probability, prediction[0][0]


# 需要预测的一些句子
sentences = [
    "<script>alert('xss')</script>",
    "hello world",
    "onclick=alert('xss')",
    "javascript:alert('xss')",
    "<script>document.cookies</script>",
    "?num=<img src=x onerror=alert('XSS')>",
    "<script src='https://www.example.com/test.js'>",
    "onclick=",
    "woc",
    "this is a document",
    "i like cookies",
]

from colorama import Fore, Back, Style

for sentence in sentences:
    probability, prediction = predict_single_sentence(sentence)
    if prediction == 0:
        output = f"{Fore.RED}{sentence}{Style.RESET_ALL} may not xss, probability of xss {Fore.RED}{str(probability[0][0])}"
        print(output)
    else:
        output = f"{Fore.RED}{sentence}{Style.RESET_ALL} may be xss, probability of xss {Fore.RED}{str(probability[0][0])}"
        print(output)
    print(Style.RESET_ALL)
