#!/usr/bin/env python
# coding: utf-8

# ## Import Necessary Modules

# In[71]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Loading DataSet

# In[72]:


df= pd.read_csv("wine_quality.csv")
df


# In[73]:


df.columns


# In[74]:


df.shape


# In[75]:


df.head()


# In[76]:


df.describe()


# In[77]:


df.corr()


# In[78]:


df.info()


# # Data Cleaning

# ## Handling Missing Values

# In[79]:


df.isnull().sum()


# In[80]:


df.groupby('quality').mean()


# # Data Visualization

# In[81]:


sns.countplot(df['quality'])
plt.show()


# In[82]:


sns.countplot(df['pH'])
plt.show()


# In[83]:


sns.countplot(df['alcohol'])
plt.show()


# In[84]:


sns.countplot(df['fixed acidity'])
plt.show()


# In[85]:


sns.countplot(df['volatile acidity'])
plt.show()


# In[86]:


sns.countplot(df['citric acid'])
plt.show()


# In[87]:


sns.countplot(df['density'])
plt.show()


# In[88]:


sns.kdeplot(df.query('quality > 2').quality)


# In[89]:


sns.distplot(df['alcohol'])


# In[90]:


df.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)


# ## Handling Outliers

# In[91]:


def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), df[column].median(), df[column])


# Apply the outlier handling function to each numerical column
for column in numerical_columns:
    handle_outliers_iqr(df, column)


# In[92]:


df.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)


# In[93]:


df.hist(figsize=(10,10),bins=50)
plt.show()


# ## Correlation

# In[94]:


corr = df.corr()
sns.heatmap(corr,annot=True)


# In[95]:


sns.pairplot(df)


# In[96]:


sns.violinplot(x='quality', y='alcohol', data=df)


# In[97]:


df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
X = df.drop(['quality','goodquality'], axis = 1)
Y = df['goodquality']


# In[98]:


df['goodquality'].value_counts()


# In[99]:


df.shape


# In[100]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print(score)


# ## Train_Test_Split

# In[101]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)


# ## Logistic Regression

# In[102]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))


# In[103]:


confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)


# ## KNN

# In[104]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))


# ## SVM

# In[105]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train,Y_train)
pred_y = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,pred_y))


# ## Decision Tree

# In[106]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))


# ## Naive Bayes

# In[107]:


from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred3))


# ## Random Forest

# In[108]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred2))


# ## XG BOOST

# In[109]:


import xgboost as xgb
model5 = xgb.XGBClassifier(random_state=1)
model5.fit(X_train, Y_train)
y_pred5 = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred5))


# In[110]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','KNN', 'SVC','Decision Tree' ,'GaussianNB','Random Forest','Xgboost'],
    'Score': [0.871,0.819,0.874,0.854,0.854,0.906,0.903]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df

