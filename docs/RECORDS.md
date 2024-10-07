# 记录

### 2024/10/06

* train_test_split
  * test_size=0.2
  * random_state=42
* Word2Vec
  * vector_size=100
  * window=5
  * min_count=1
  * workers=multiprocessing.cpu_count()
* SVC
  * kernel='linear'

```text
Accuracy: 52.32%
```

### 2024/10/07

* train_test_split
  * test_size=0.3
  * random_state=42
* Word2Vec
  * vector_size=100
  * window=5
  * min_count=1
  * workers=multiprocessing.cpu_count()
* SVC
  * C=100
  * probability=True

```text
                precision    recall  f1-score   support
         伤心       0.50      0.86      0.64      4217
         害怕       0.62      0.09      0.15       209
         开心       0.65      0.59      0.62      2966
         恶心       0.68      0.16      0.26      1473
         惊喜       0.00      0.00      0.00       311
         生气       0.48      0.09      0.14      1328

    accuracy                           0.55     10504
   macro avg       0.49      0.30      0.30     10504
weighted avg       0.56      0.55      0.49     10504

Accuracy: 54.74%
```
