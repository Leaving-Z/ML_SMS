# ML_SMS
2025年春机器学习大作业

## 依赖

`pip install pandas nltk scikit-learn`  

在第一次运行时，需要在代码开头执行  
```python
nltk.download('punkt')
nltk.download('stopwords')
```
加载停止词

## 训练和结果查看

执行 `python compare.py`，预期结果大致如下  
```
=== Model Performance (Classical Models) ===
MultinomialNB      | Acc 0.9839 | Prec 0.9925 | Rec 0.8859 | F1 0.9362
LogisticRegression | Acc 0.9821 | Prec 0.9448 | Rec 0.9195 | F1 0.9320
LinearSVC          | Acc 0.9830 | Prec 0.9779 | Rec 0.8926 | F1 0.9333
RandomForest       | Acc 0.9776 | Prec 1.0000 | Rec 0.8322 | F1 0.9084
KNN                | Acc 0.9085 | Prec 1.0000 | Rec 0.3154 | F1 0.4796
```