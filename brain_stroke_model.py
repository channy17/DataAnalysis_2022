import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path = 'yourfilepathiniput'
df = pd.read_csv(path+'/brain_stroke.csv')
#print(df.info())

df.loc[df['gender']=='Female', 'gender'] = 0
df.loc[df['gender']=='Male', 'gender'] = 1
df['gender'] = df['gender'].astype('int')

#print(df.info())

sel = ['hypertension', 'heart_disease', 'age']
X = df[sel]
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77, stratify=y)

from sklearn.ensemble import GradientBoostingClassifier

model_gbc_base = GradientBoostingClassifier().fit(X_train,y_train)
print('base_train : ', model_gbc_base.score(X_train, y_train))
print('base_test : ', model_gbc_base.score(X_test, y_test))

# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder

sel_enc =['ever_married', 'work_type', 'Residence_type', 'smoking_status']
en_x = LabelEncoder()
sel_enc_n = []

for i in range(0,4) :
 df[sel_enc[i]+'_lbl'] = en_x.fit_transform(df[sel_enc[i]])
 sel_enc_n.append(sel_enc[i]+'_lbl')
 

df_n = df.drop(sel_enc, axis=1)
#print(df_n.head(3))
sel_total = sel + sel_enc_n

X_tr = df[sel_total]
X_train, X_test, y_train, y_test = train_test_split(X_tr, y, stratify=y, random_state=77)

model_gbc = GradientBoostingClassifier().fit(X_train,y_train)
print('final_train : ', model_gbc.score(X_train, y_train))
print('final_test : ', model_gbc.score(X_test, y_test))

