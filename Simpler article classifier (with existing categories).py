

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics




df = pd.read_csv('bbc-text.csv')

fig = px.histogram(df, x="category")



blanks = []  

for i,lb,rv in df.itertuples(): 
    if type(rv)==str:            
        if rv.isspace():       
            blanks.append(i)     
        
print(len(blanks), 'blanks: ', blanks)



X = df['text']  
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)



text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])



text_clf.fit(X_train, y_train)  

prediction = text_clf.predict(X_test)

cm = metrics.confusion_matrix(y_test,prediction)

fig = px.imshow(cm)
fig.show()

print(metrics.classification_report(y_test,prediction))

print(metrics.accuracy_score(y_test,prediction))


newarticle = "Rafael dos Anjos dispatched Paul Felder at UFC Vegas 14 and immediately called out Conor McGregor for his next fight.  After his points win over five rounds, the Brazilian said the lightweight division was wide open and called out the Irishman, another former champion. 'I think me and Conor is the fight to make', said the 36-year-old."


print(text_clf.predict([newarticle]))


newarticle2 = "Recently, a video of a young man taking his own life was posted on Facebook. The footage subsequently spread to other platforms, including TikTok, where it stayed online for days. TikTok has acknowledged users would be better protected if social media providers worked more closely together. But Ruth echoes the NSPCC's view and thinks social networks should not be allowed to police themselves. She says some of the material her daughter accessed six years ago is still online, and typing certain words into Facebook or Instagram brings up the same imagery. Facebook announced the expansion of an automated tool to recognise and remove self-harm and suicide content from Instagram earlier this week, but has said data privacy laws in Europe limit what it can do. Other smaller start-ups are also trying to use technology to address the issue. SafeToWatch is developing software that is trained by machine-learning techniques to block inappropriate scenes including violence and nudity in real-time."



print(text_clf.predict([newarticle2]))




