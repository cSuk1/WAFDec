# coding=utf-8
import fileinput
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sys
import re
import nltk
import sklearn
import tensorflow.keras as keras
import tensorflow.keras.preprocessing as keras_preprocessing
from sklearn.preprocessing import StandardScaler
import chardet
import math
from joblib import dump

g_word_dict = {}
g_keras_token = None


# 定义一个函数，用于查找指定文件夹中指定后缀名的文件
def os_listdir_ex(file_dir, find_name):
    # 定义一个空列表，用于存放查找结果
    result = []
    # 遍历指定文件夹及其子文件夹中的文件
    for root, dirs, files in os.walk(file_dir):
        # 遍历文件
        for file in files:
            # 如果文件的后缀名为指定后缀名，则将文件路径添加到结果列表中
            if os.path.splitext(file)[1] == find_name:
                result.append(os.path.join(root, file))
    # 返回查找结果
    return result


# 获取文件长度
def get_file_length(pFile):
    # 获取文件大小
    fsize = os.path.getsize(pFile)
    # 返回文件大小
    return int(fsize)


# 定义函数get_data_frame()，返回一个DataFrame，其中包含标签和文件名
def get_data_frame():  # 得到data frame
    # 得到webshell列表
    webshell_files = os_listdir_ex(".\\data\\washedwebshell", ".php")
    # 得到正常文件列表
    normal_files = os_listdir_ex(".\\data\\washedwordpress", ".php")
    label_webshell = []
    label_normal = []
    # 打上标注
    for i in range(0, len(webshell_files)):
        label_webshell.append(1)
    for i in range(0, len(normal_files)):
        label_normal.append(0)
    # 合并起来
    files_list = webshell_files + normal_files
    label_list = label_webshell + label_normal
    state = np.random.get_state()
    np.random.shuffle(files_list)  # 训练集
    np.random.set_state(state)
    np.random.shuffle(label_list)  # 标签

    data_list = {"label": label_list, "file": files_list}
    return pd.DataFrame(data_list, columns=["label", "file"])


# 定义一个函数，用于清洗php注释
def flush_file(pFile):  # 清洗php注释
    # 打开文件，以gb18030编码，忽略错误
    file = open(pFile, "r", encoding="gb18030", errors="ignore")
    # 读取文件内容
    read_string = file.read()
    # 关闭文件
    file.close()
    # 使用正则表达式匹配/*.*?*/，替换为空字符
    m = re.compile(r"/\*.*?\*/", re.S)
    result = re.sub(m, "", read_string)
    # 使用正则表达式匹配//.*，替换为空字符
    m = re.compile(r"//.*")
    result = re.sub(m, "", result)
    # 使用正则表达式匹配#.*，替换为空字符
    m = re.compile(r"#.*")
    result = re.sub(m, "", result)
    # 返回替换后的字符串
    return result


# 定义函数get_file_entropy，用于计算文件的熵
# 参数pFile为文件路径
def get_file_entropy(pFile):
    # 调用flush_file函数，将文件内容写入clean_string
    clean_string = flush_file(pFile)
    # 创建一个字典，用于存储文件中的字符
    text_list = {}
    # 初始化变量_sum和result
    _sum = 0
    result = 0
    # 遍历clean_string中的每一个字符
    for word_iter in clean_string:
        # 如果字符不是换行符和空格，则将其加入字典
        if word_iter != "\n" and word_iter != " ":
            if word_iter not in text_list.keys():
                text_list[word_iter] = 1
            else:
                text_list[word_iter] = text_list[word_iter] + 1
    # 遍历字典中的每一个元素，计算_sum
    for index in text_list.keys():
        _sum = _sum + text_list[index]
    # 遍历字典中的每一个元素，计算result
    for index in text_list.keys():
        result = result - float(text_list[index]) / _sum * math.log(
            float(text_list[index]) / _sum, 2
        )
    # 返回result
    return result


# 定义一个函数，用于将序列向量化，参数sequences为序列，dimention为维度，默认为1337
def vectorize_sequences(sequences, dimention=1337):
    # 创建一个大小为（25000，10000）的全零矩阵
    results = np.zeros((len(sequences), dimention))
    # 遍历sequences中的每一个序列
    for i, sequence in enumerate(sequences):
        # 如果i大于dimention，则跳出循环
        if i > dimention:
            break
        try:
            # 将sequence中的值设置为1.0
            results[i, sequence] = 1.0
        except:
            # 如果出现异常，则跳出循环
            break

    return results


# 定义函数get_word_bag，用于获取单词包
# 参数pWordList：单词列表
# 返回值word_bag：单词包
def get_word_bag(pWordList):
    """
    获取单词包
    :param pWordList: 单词列表
    :return: word_bag: 单词包
    """
    global g_word_dict
    global g_keras_token
    sequences_data = g_keras_token.texts_to_sequences(pWordList)
    word_bag = []
    for index in range(0, len(sequences_data)):
        if len(sequences_data[index]) != 0:
            for zeus in range(0, len(sequences_data[index])):
                word_bag.append(sequences_data[index][zeus])
    return word_bag


# 设置单词包，pWordList为单词列表
def set_word_bag(pWordList):
    global g_word_dict
    global g_keras_token
    if g_keras_token == None:
        g_keras_token = keras.preprocessing.text.Tokenizer()  # 初始化标注器
    g_keras_token.fit_on_texts(pWordList)  # 学习出文本的字典
    g_word_dict.update(g_keras_token.word_index)


# 定义一个函数，用于获取文件中的单词，参数为文件路径
def get_file_word(pFile):
    # 声明全局变量g_word_dict
    global g_word_dict
    # 定义英文标点符号
    english_punctuations = [
        ",",
        ".",
        ":",
        ";",
        "?",
        "(",
        ")",
        "[",
        "]",
        "&",
        "!",
        "*",
        "@",
        "#",
        "$",
        "%",
        "php",
        "<",
        ">",
        "'",
    ]
    # 调用flush_file函数，获取文件内容
    clean_string = flush_file(pFile)
    # 使用nltk.word_tokenize函数，将文件内容分割成单词列表
    word_list = nltk.word_tokenize(clean_string)
    word_list = [
        word_iter for word_iter in word_list if word_iter not in english_punctuations
    ]
    # anti-paste
    # 反粘贴
    return word_list


# 定义一个函数，用于获取文件中的单词包，不使用停用词
def get_file_word_bag_non_use(pFile):
    # 声明全局变量g_word_dict
    global g_word_dict
    # 定义英文标点符号
    english_punctuations = [
        ",",
        ".",
        ":",
        ";",
        "?",
        "(",
        ")",
        "[",
        "]",
        "&",
        "!",
        "*",
        "@",
        "#",
        "$",
        "%",
        "php",
        "<",
        ">",
        "'",
    ]
    # 调用flush_file函数，获取文件内容
    clean_string = flush_file(pFile)
    # 使用nltk.word_tokenize函数，将文件内容分割成单词列表
    word_list = nltk.word_tokenize(clean_string)
    # 将单词列表中，不在english_punctuations中的单词添加到word_list中
    word_list = [
        word_iter for word_iter in word_list if word_iter not in english_punctuations
    ]

    # 初始化标注器
    keras_token = keras.preprocessing.text.Tokenizer()  # 初始化标注器
    # 学习出文本的字典
    keras_token.fit_on_texts(word_list)  # 学习出文本的字典
    # 将字典中的单词添加到全局变量g_word_dict中
    g_word_dict.update(keras_token.word_index)
    # 通过texts_to_sequences 这个dict可以将每个string的每个词转成数字
    sequences_data = keras_token.texts_to_sequences(word_list)
    word_bag = []
    # 遍历sequences_data，将每个单词的每个词添加到word_bag中
    for index in range(0, len(sequences_data)):
        if len(sequences_data[index]) != 0:
            for zeus in range(0, len(sequences_data[index])):
                word_bag.append(sequences_data[index][zeus])
    # 返回word_bag
    return word_bag


# 定义构建网络的函数
def build_network():
    # 声明全局变量
    global g_word_dict
    # 定义输入层，输入维度为1337，数据类型为int16，名称为word_bag
    input_1 = keras.layers.Input(shape=(1337,), dtype="int16", name="word_bag")
    # 词嵌入（使用预训练的词向量）
    embed = keras.layers.Embedding(len(g_word_dict) + 1, 300, input_length=1337)(
        input_1
    )
    # 词窗大小分别为3,4,5
    cnn1 = keras.layers.Conv1D(256, 3, padding="same", strides=1, activation="relu")(
        embed
    )
    cnn1 = keras.layers.MaxPooling1D(pool_size=48)(cnn1)

    cnn2 = keras.layers.Conv1D(256, 4, padding="same", strides=1, activation="relu")(
        embed
    )
    cnn2 = keras.layers.MaxPooling1D(pool_size=47)(cnn2)

    cnn3 = keras.layers.Conv1D(256, 5, padding="same", strides=1, activation="relu")(
        embed
    )
    cnn3 = keras.layers.MaxPooling1D(pool_size=46)(cnn3)
    # 合并三个模型的输出向量
    cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=1)
    flat = keras.layers.Flatten()(cnn)
    drop = keras.layers.Dropout(0.2)(flat)
    # 定义第一个输出层，输出维度为1，激活函数为sigmoid，名称为TextCNNoutPut
    model_1_output = keras.layers.Dense(1, activation="sigmoid", name="TextCNNoutPut")(
        drop
    )

    # 第二层
    # 定义输入层，输入维度为2，数据类型为float32，名称为length_entropy
    input_2 = keras.layers.Input(shape=(2,), dtype="float32", name="length_entropy")
    # 定义第一个隐藏层，输入维度为2，激活函数为relu
    model_2 = keras.layers.Dense(128, input_shape=(2,), activation="relu")(input_2)
    # 定义第一个隐藏层，输入维度为2，激活函数为relu，dropout比例为0.4
    model_2 = keras.layers.Dropout(0.4)(model_2)
    # 定义第二个隐藏层，输入维度为128，激活函数为relu，dropout比例为0.2
    model_2 = keras.layers.Dense(64, activation="relu")(model_2)
    # 定义第二个隐藏层，输入维度为64，激活函数为relu，dropout比例为0.2
    model_2 = keras.layers.Dropout(0.2)(model_2)
    # 定义第三个隐藏层，输入维度为64，激活函数为relu，dropout比例为0.2
    model_2 = keras.layers.Dense(32, activation="relu")(model_2)
    # 定义第二个输出层，输出维度为1，激活函数为sigmoid，名称为LengthEntropyOutPut
    model_2_output = keras.layers.Dense(
        1, activation="sigmoid", name="LengthEntropyOutPut"
    )(model_2)

    # 将辅助输入数据与TextCNN层的输出连接起来,输入到模型中
    model_combined = keras.layers.concatenate([model_2_output, model_1_output])
    # 定义输出层，输出维度为1，激活函数为sigmoid，名称为main_output
    model_end = keras.layers.Dense(64, activation="relu")(model_combined)
    model_end = keras.layers.Dense(1, activation="sigmoid", name="main_output")(
        model_end
    )

    # 定义这个具有两个输入和输出的模型
    model_end = keras.Model(inputs=[input_2, input_1], outputs=model_end)
    # 编译模型，优化器为adam，损失函数为binary_crossentropy，指标为accuracy
    model_end.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    # 返回模型
    return model_end


# 定义一个函数，用于获取数据框，如果save.csv文件不存在，则获取数据框，并将其保存到save.csv文件中，如果save.csv文件已存在，则从save.csv文件中读取数据框
data_frame = []
if os.path.exists("save.csv") == False:
    # 获取数据框
    data_frame = get_data_frame()
    # 打印数据框的前5行
    print(data_frame.head(5))
    # 计算文件长度，并将其保存到data_frame中
    data_frame["length"] = (
        data_frame["file"].map(lambda file_name: get_file_length(file_name)).astype(int)
    )
    # 计算文件熵，并将其保存到data_frame中
    data_frame["entropy"] = (
        data_frame["file"]
        .map(lambda file_name: get_file_entropy(file_name))
        .astype(float)
    )
    # 初始化标准化器，用于标准化文件长度和文件熵
    scaler_length = StandardScaler()
    scaler_entropy = StandardScaler()
    # 标准化文件长度和文件熵
    data_frame["length_scaled"] = scaler_length.fit_transform(
        data_frame["length"].values.reshape(-1, 1),
        scaler_length.fit(data_frame["length"].values.reshape(-1, 1)),
    )
    data_frame["entropy_scaled"] = scaler_entropy.fit_transform(
        data_frame["entropy"].values.reshape(-1, 1),
        scaler_entropy.fit(data_frame["entropy"].values.reshape(-1, 1)),
    )
    # 获取文件中的单词，并将其保存到data_frame中
    data_frame["word_bag"] = data_frame["file"].map(
        lambda file_name: get_file_word(file_name)
    )
    # 设置单词袋，并将其保存到data_frame中
    set_word_bag(data_frame["word_bag"])
    # 获取单词袋，并将其保存到data_frame中
    data_frame["word_bag"] = data_frame["word_bag"].map(
        lambda text_params: get_word_bag(text_params)
    )

    # 将data_frame保存到save.csv文件中
    data_frame.to_csv("save.csv")
    # 将标准化器scaler_length保存到webshell-models/scaler_length.joblib文件中
    dump(scaler_length, "webshell-models/scaler_length.joblib")
    # 将标准化器scaler_entropy保存到webshell-models/scaler_entropy.joblib文件中
    dump(scaler_entropy, "webshell-models/scaler_entropy.joblib")
else:
    # 从save.csv文件中读取数据框
    data_frame = pd.read_csv("save.csv", header=0)
    # 打印数据框的前5行
    print(data_frame.head(5))
# 跳过数据
skip_Data_num = 2610
# 从data_frame中获取训练集x_1和x_2，并将其转换为常量
data_train_pre = data_frame.filter(items=["length_scaled", "entropy_scaled"])
data_train_y = tf.constant(data_frame.filter(items=["label"])[:skip_Data_num])

# 加载数据集，跳过前skip_Data_num个数据
data_train_x_1 = tf.constant(data_train_pre[:skip_Data_num])
# 将data_frame["word_bag"].values转换为向量，跳过前skip_Data_num个数据
data_train_x_2 = tf.constant(
    vectorize_sequences(data_frame["word_bag"].values[:skip_Data_num])
)
# 打印data_train_x_1和data_train_x_2
print(data_train_x_1, data_train_x_2)

# 构建神经网络模型
network_model = build_network()
# 打印模型概述
network_model.summary()
# 使用训练数据集训练模型，训练200次
history = network_model.fit(
    x=[data_train_x_1, data_train_x_2], y=data_train_y, epochs=200
)
# 保存模型
network_model.save("webshell-models/webshell.h5")
