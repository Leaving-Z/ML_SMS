# model_training.py
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from data_process import load_data, preprocess_data, extract_features

# 1) 数据加载与处理
df = load_data("data/SMSSpamCollection")
df = preprocess_data(df)

# 2) 特征提取
X, y = extract_features(df)

# 3) 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4) 模型训练
clf = LinearSVC()
clf.fit(X_train, y_train)

# 5) 评估模型
y_pred = clf.predict(X_test)
print("Acc:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
