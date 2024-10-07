# -*- coding: utf-8 -*-

from joblib import dump, load
from tqdm import tqdm
import json
import multiprocessing
import logging
import time
import jieba
from gensim.models import Word2Vec
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(level=logging.INFO)

# 停用词
stop_words = set(open('dicts/chinese_stopwords.txt', 'r', encoding='utf-8').read().split())


# 原始语料内容转换为数组
def convert_to_sentiment_list():
    sentiment_map = {
        '0': '开心',
        '1': '伤心',
        '2': '恶心',
        '3': '生气',
        '4': '害怕',
        '5': '惊喜',
    }

    result = []

    with open('origin/data.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                label, text = line.split(' ', 1)
                sentiment = sentiment_map.get(label)

                if sentiment:
                    result.append((text, sentiment))

    return result


# 1. 中文文本预处理
def preprocess_chinese(text):
    # 中文分词
    words = jieba.cut(text, cut_all=False)

    # 去停用词
    words = [word for word in words if word not in stop_words]

    return words[:50]


# 2.训练Word2Vec模型
def word2vec(data):
    sentences = [text for text, label in data]

    model = Word2Vec(
        sentences,
        vector_size=100,
        window=8,
        min_count=2,
        sg=1,
        negative=10,
        epochs=30,
        workers=multiprocessing.cpu_count(),
    )

    model.save('models/word2vec.model')


# 3.将中文文本转换为向量
def get_sentence_vector(sentence, model):
    # 获取句子的词向量，取词向量的平均值
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


# 4.训练 SVM 模型
def trainSVM(X, y):
    # 划分训练集和测试集
    logging.info('SVM: Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练 SVM 分类器
    logging.info('SVM: Training SVM model...')
    svm_model = SVC(C=100, probability=True)
    svm_model.fit(X_train, y_train)

    # 预测测试集
    logging.info("SVM: Predicting on test data...")
    y_pred = svm_model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(classification_report(y_test, y_pred))
    logging.info(f"Accuracy: {accuracy * 100:.2f}%")

    # 保存模型
    logging.info("SVM: Saving model...")
    dump(svm_model, 'models/svm.joblib')


# 5.生成中文情感词典
def generate_sentiment_dictionary(word2vec_model, svm_model):
    sentiment_dict = {}
    for word in tqdm(word2vec_model.wv.index_to_key):
        vector = word2vec_model.wv[word].reshape(1, -1)
        sentiment_label = svm_model.predict(vector)[0]
        sentiment_dict[word] = sentiment_label
    return sentiment_dict


dataList = convert_to_sentiment_list()
preprocessed_data = [(preprocess_chinese(text), label) for text, label in dataList]
word2vec(preprocessed_data)
time.sleep(2)

# 读取word2vec模型
word2vecModel = Word2Vec.load('models/word2vec.model')

time.sleep(2)

# 对所有中文文本进行向量化
X = np.array([
    get_sentence_vector(
        text,
        word2vecModel,
    ) for text, label in tqdm(preprocessed_data)
])
y = np.array([label for text, label in preprocessed_data])

time.sleep(2)

trainSVM(X, y)

time.sleep(2)

# 读取svm模型
svmModel = load('models/svm.joblib')

time.sleep(2)

# 生成情感词典
sentiment_dict = generate_sentiment_dictionary(word2vecModel, svmModel)

# 保存词典
with open('models/sentiment_dict.json', 'w', encoding='utf-8') as f:
    json.dump(sentiment_dict, f, ensure_ascii=False, indent=4)
