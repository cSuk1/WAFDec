from collections import Counter
import re
from urllib.parse import unquote
from gensim.models.word2vec import Word2Vec
import pandas as pd
import nltk


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


# 读取普通数据，并将其命名为payload，frac=0.1表示从数据集中随机抽取10%的数据
normal_data = pd.read_csv("sql-inject/benign.csv", names=["payload"]).sample(frac=0.1)
# 读取恶意数据，并将其命名为payload，frac=0.1表示从数据集中随机抽取10%的数据
sql_data = pd.read_table("sql-inject/malicious.csv", names=["payload"], sep="~").sample(
    frac=0.1
)


normal_data["label"] = 0
sql_data["label"] = 1

# 将正常数据和sql攻击数据按行合并
data = pd.concat([normal_data, sql_data])
# 将data中的payload列映射为GeneSeg函数，并将结果赋值给words列
data["words"] = data["payload"].map(GeneSeg)

# 展开词汇列表
all_words = [word for sublist in data[data["label"] == 1]["words"] for word in sublist]

# 统计词频
word_counts = Counter(all_words)
vocabulary_size = 3000
# 选取出现次数最多的3000个词汇构建词汇表
top_words = [word for word, count in word_counts.most_common(vocabulary_size)]

# 构建词汇表
vocab = {word: idx for idx, word in enumerate(top_words)}  # 构建词汇表

# 根据词汇表替换词汇列表中的词汇
processed_words = [
    [word if word in vocab else "UNK" for word in sublist]
    for sublist in data[data["label"] == 1]["words"]
]

embedding_size = 300
num_sampled = 20
skip_window = 10
num_iter = 10

model = Word2Vec(
    processed_words,
    vector_size=embedding_size,
    window=skip_window,
    negative=num_sampled,
    epochs=num_iter,
)
embeddings = model.wv
import os

# 创建保存模型的文件夹
save_dir = "sql-models"
os.makedirs(save_dir, exist_ok=True)
# 保存模型
model.save(os.path.join(save_dir, "trained_w2v_model.h5"))

# embeddings.similar_by_word("select", 10)

import numpy as np
import tensorflow as tf

# 准备数据
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data["words"])

import pickle

# 保存 Tokenizer 对象
with open("sql-models/tokenizer.pickle", "wb") as file:
    pickle.dump(tokenizer, file)

word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(data["words"])
X = pad_sequences(X)
print("max length" + str(X.shape[1]))

# 处理标签
Y = np.array(data["label"])

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense

# 构建LSTM模型
lstm_model = Sequential()
lstm_model.add(Embedding(len(word_index) + 1, 128, input_length=X.shape[1]))
lstm_model.add(SpatialDropout1D(0.2))
lstm_model.add(LSTM(100))
lstm_model.add(Dense(1, activation="tanh"))

# 编译模型
lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练模型
lstm_model.fit(
    X_train, Y_train, epochs=3, batch_size=16, validation_data=(X_test, Y_test)
)

# 创建保存模型的文件夹
os.makedirs(save_dir, exist_ok=True)
# 保存模型
lstm_model.save(os.path.join(save_dir, "trained_sqldec_model.h5"))
# 进行预测
y_pred = lstm_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# 进行评估
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")
