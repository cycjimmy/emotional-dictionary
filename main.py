# -*- coding: utf-8 -*-

from joblib import dump, load
from tqdm import tqdm
import json
import multiprocessing
import logging
import time
from gensim.models import Word2Vec
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from utils import convert_to_sentiment_list, preprocess_chinese

logging.basicConfig(level=logging.INFO)

# 训练Word2Vec模型
def word2vec(text):
    sentences = [text_.split() for text_ in text]
    word2vec_model = Word2Vec(
        sentences,
        vector_size=100,
        window=6,
        min_count=1,
        sg=1,
        epochs=30,
        workers=multiprocessing.cpu_count(),
    )

    word2vec_model.save('models/word2vec.model')

# 转换为句子向量
def sentence_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# 训练 SVM 模型
def trainSVM(X, y):
    # 划分训练集和测试集
    logging.info('SVM: Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义SVM参数网格
    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')

    # 训练并寻找最佳参数
    logging.info('SVM: Training and finding the best parameters...')
    grid_search.fit(X_train, y_train)
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    logging.info(f"Best Accuracy: {grid_search.best_score_}")

    # 使用最佳参数重新训练SVM模型
    best_classifier = grid_search.best_estimator_
    y_pred = best_classifier.predict(X_test)
    logging.info(classification_report(y_test, y_pred))

    # 保存模型
    logging.info("SVM: Saving model...")
    dump(best_classifier, 'models/svm.joblib')


# 生成中文情感词典
def generate_sentiment_dictionary(word2vec_model, svm_model):
    sentiment_dict = {}
    for word in tqdm(word2vec_model.wv.index_to_key):
        vector = word2vec_model.wv[word].reshape(1, -1)
        sentiment = svm_model.predict(vector)[0]
        sentiment_dict[word] = sentiment
    return sentiment_dict


dataList = convert_to_sentiment_list()
preprocessed_data = [(preprocess_chinese(text), label) for text, label in dataList]

# 分离文本和标签
preprocessed_text = [sub_data[0] for sub_data in preprocessed_data]
preprocessed_label = [sub_data[1] for sub_data in preprocessed_data]

# 训练Word2Vec模型
word2vec(preprocessed_text)
time.sleep(2)

# 读取Word2Vec模型
word2vec_model = Word2Vec.load('models/word2vec.model')

# 转换为句子向量
X_vec = np.array([sentence_vector(sentence, word2vec_model) for sentence in preprocessed_text])

# 训练SVM模型
trainSVM(X_vec, preprocessed_label)

# 读取svm模型
svm_model = load('models/svm.joblib')
time.sleep(2)

# 生成情感词典
sentiment_dict = generate_sentiment_dictionary(word2vec_model, svm_model)

# 保存词典
with open('models/sentiment_dict.json', 'w', encoding='utf-8') as f:
    json.dump(sentiment_dict, f, ensure_ascii=False, indent=4)
