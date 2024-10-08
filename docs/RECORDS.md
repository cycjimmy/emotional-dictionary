# 记录

## 当前
### #1 2024/10/08

* train_test_split
  * test_size=0.2
  * random_state=42
* Word2Vec
  * vector_size=100
  * window=6
  * min_count=1
  * sg=1
  * epochs=30
  * workers=multiprocessing.cpu_count()
* SVC(搜索模式)
  * C=1
  * kernel='rbf'

```text
                precision    recall  f1-score   support
         伤心       0.55      0.83      0.66      2802
         害怕       0.79      0.17      0.28       133
         开心       0.65      0.68      0.66      2004
         恶心       0.70      0.21      0.33      1005
         惊喜       0.75      0.01      0.03       206
         生气       0.50      0.19      0.28       853

    accuracy                           0.58      7003
   macro avg       0.66      0.35      0.37      7003
weighted avg       0.60      0.58      0.54      7003

Accuracy: 58.24%
```

## 归档
* [word2vec_svc.1.md](word2vec_svc.1.md)
