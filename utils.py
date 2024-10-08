import jieba

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


# 中文文本预处理
def preprocess_chinese(text):
    # 中文分词
    words = jieba.cut(text, cut_all=False)

    # 去停用词
    words = [word for word in words if word not in stop_words]

    return ' '.join(words[:50])
