

# Copy and paste the entire backend code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk as nlp
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")
df_fake.head()


df_true["text"] = df_true["text"].replace("(Reuters)","",regex=True)
df_true.head()
## The String "()" still remains in the text column to be removed which i take care of during the NLP part done below


df_fake["target"] = 0
df_true["target"] = 1


df_true.head(10)


df_fake.head(10)




df_fake = df_fake.drop(["title","subject","date"],axis = 1)
df_true = df_true.drop(["title","subject","date"],axis = 1)



df = pd.concat([df_fake,df_true],axis = 0)


df.head(10)



df = df.sample(frac=1)
df.head(10)

df.reset_index(inplace=True)
df.drop(["index"], axis = 1, inplace = True)
df.head(10)


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    # This is where i remove the "()" from the text column. You can do in whatever way you want 
    # The key is to remove the "(Reuters)" string as it is present in all text of True.csv.
    # The Model during the training part can memorize it and perfrom great in training and badly when other testing input is given.
    text = re.sub('[()]','',text)
    text = re.sub('\\W',' ',text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text



df["text"] = df["text"].apply(wordopt)
df.head(100)



X = df["text"]
Y = df["target"]
X.shape


X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size=0.25)
X_train.shape




from sklearn.feature_extraction.text import TfidfVectorizer
print(X_train)
vectorization = TfidfVectorizer()
analyze = vectorization.build_analyzer()
print(analyze(X_train[0]))
xv_train = vectorization.fit_transform(X_train)
xv_test = vectorization.transform(x_test)
#print(xv_train.shape)
#print(xv_test.shape)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(xv_train,Y_train)
lr.score(xv_test,y_test)
y_test,lr.predict(xv_test)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(xv_train,Y_train)
xv_test,y_test
y_test,dtc.predict(xv_test)


from sklearn.ensemble import GradientBoostingClassifier
gclf = GradientBoostingClassifier()
gclf.fit(xv_train,Y_train)
gclf.score(xv_test,y_test)
y_test,gclf.predict(xv_test)


from sklearn.ensemble import RandomForestClassifier
rclf = RandomForestClassifier()
rclf.fit(xv_train,Y_train)
rclf.score(xv_test,y_test)
y_test,rclf.predict(xv_test)




def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    


def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = lr.predict(new_xv_test)
    pred_DT = dtc.predict(new_xv_test)
    pred_GBC = gclf.predict(new_xv_test)
    pred_RFC = rclf.predict(new_xv_test)

    result = "\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),
                                                                                                           output_lable(pred_DT[0]), 
                                                                                                           output_lable(pred_GBC[0]), 
                                                                                                           output_lable(pred_RFC[0]))
    return result