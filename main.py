# -*- coding: utf-8 -*-

from joblib import dump, load
from tqdm import tqdm
import json
import logging
import time
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from utils import convert_to_sentiment_list, preprocess_chinese

logging.basicConfig(level=logging.INFO)


# 建立句子向量
def build_sentences_vectors(data, model):
    label = []
    vec = []

    # 返回特征词向量
    def getWordVecs(wordList_):
        vecs_ = []
        for word in wordList_:
            try:
                vecs_.append(model.wv[word])
            except KeyError:
                continue
        return np.array(vecs_, dtype='float')

    for item in data:
        sentence, label_ = item
        wordList = sentence.split(' ')

        vecs = getWordVecs(wordList)
        if len(vecs) > 0:
            vecsArray = sum(np.array(vecs)) / len(vecs)
            vec.append(vecsArray)
            label.append(label_)
    return vec[:], label


# 训练Word2Vec模型
def word2vec(preprocessed_data_):
    sentences = [text for text, label in preprocessed_data_]

    word2vec_model = Word2Vec(
        sentences,
        vector_size=100,
        window=6,
        min_count=1,
        sg=1,
        epochs=30,
        workers=4,
    )

    # 转换为句子向量
    X_vec, label = build_sentences_vectors(preprocessed_data_, word2vec_model)

    df_x = pd.DataFrame(X_vec)
    df_y = pd.DataFrame(label)
    data = pd.concat([df_y, df_x], axis=1)

    # 导出模型和词向量和句子向量
    word2vec_model.save('models/word2vec.model')
    word2vec_model.wv.save_word2vec_format('models/word2vec.vector', binary=False)
    data.to_csv('models/word2vec.csv')


# 训练 SVM 模型
def trainSVM():
    data = pd.read_csv('models/word2vec.csv')

    # 提取标签
    label = data.iloc[:, 1]

    # 从第三列（索引2）开始提取特征数据
    X = data.iloc[:, 2:]

    # 划分训练集和测试集
    logging.info('SVM: Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)

    # 定义SVM参数网格
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],  # 仅在 kernel='poly' 时发挥作用
        'gamma': ['scale', 'auto'],  # 控制 RBF, poly 和 sigmoid 的影响
        'coef0': [0.0, 0.1, 0.5],  # 仅在 kernel='poly' 和 'sigmoid' 时有效
        'shrinking': [True, False],  # 使用启发式收缩
        'tol': [1e-3, 1e-4]  # 停止准则的容忍度
    }
    svc = SVC(verbose=True)
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')

    # 训练并寻找最佳参数
    logging.info('SVM: Training and finding the best parameters...')
    grid_search.fit(X_train, y_train)
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    logging.info(f"Best Accuracy: {grid_search.best_score_}")

    best_classifier = grid_search.best_estimator_
    y_pred = best_classifier.predict(X_test)
    logging.info(classification_report(y_test, y_pred, zero_division=1))

    # 保存模型
    logging.info("SVM: Saving model...")
    feature_names = X_train.columns.tolist()
    dump((best_classifier, feature_names), 'models/svm_with_features.joblib')


# 生成中文情感词典
def generate_sentiment_dictionary():
    word2vec_model = Word2Vec.load('models/word2vec.model')
    svm_model, feature_names = load('models/svm_with_features.joblib')

    sentiment_dict = {}
    for word in tqdm(word2vec_model.wv.index_to_key):
        vector = word2vec_model.wv[word].reshape(1, -1)
        vector_df = pd.DataFrame(vector, columns=feature_names)
        sentiment = svm_model.predict(vector_df)[0]
        sentiment_dict[word] = sentiment

    # 保存词典
    with open('models/sentiment_dict.json', 'w', encoding='utf-8') as f:
        json.dump(sentiment_dict, f, ensure_ascii=False, indent=4)


dataList = convert_to_sentiment_list()
preprocessed_data = [(preprocess_chinese(text), label) for text, label in dataList]

# 训练Word2Vec模型
word2vec(preprocessed_data)
time.sleep(2)

# 训练SVM模型
trainSVM()

# 生成情感词典
generate_sentiment_dictionary()
