# model_training.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from data_process import load_data, preprocess_data, extract_features

# 1) 数据加载与处理
df = load_data("data/SMSSpamCollection")   # 修改为你的路径
df = preprocess_data(df)

# 2) 特征提取
X, y = extract_features(df)

# 3) 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4) 选择模型
models = {
    "MultinomialNB"     : MultinomialNB(alpha=0.3),
    "LogisticRegression" : LogisticRegression(max_iter=1000, class_weight='balanced'),
    "LinearSVC"         : LinearSVC(C=1.0),
    "RandomForest"      : RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
}

# 5) 定义评估函数
def evaluate(clf, name):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)
    print(f"{name:18s} | Acc {acc:.4f} | Prec {p:.4f} | Rec {r:.4f} | F1 {f:.4f}")

# 6) 输出评估结果
print("\n=== Model Performance ===")
for name, clf in models.items():
    evaluate(clf, name)
