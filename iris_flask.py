import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

path = 'yourfilepath'
df = pd.read_csv(path+'/data/iris.csv')
#print(df.shape)
#print(df.columns)
#print(df.info())

# 라벨 인코딩
df.loc[ df['Species']=='Iris-setosa', 'target'] = 0
df.loc[ df['Species']=='Iris-versicolor', 'target'] = 1
df.loc[ df['Species']=='Iris-virginica', 'target'] = 2
df['target'] = df['target'].astype("int")

#print(df.head())

# 데이터 선택 및 나누기
sel = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[sel]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 모델 생성 및 평가
model_rf = GradientBoostingClassifier(random_state=30).fit(X_train, y_train)
scores = cross_val_score(model_rf, X_test, y_test, cv=5, scoring='accuracy')
#print("교차 검증 점수(AUC) - 그래디언트 부스팅 : ", scores.mean())

# pkl(모델) 파일로 저장하기

pickle.dump(model_rf, open(path + "\\ml_model\\iris_base.pkl", 'wb'))