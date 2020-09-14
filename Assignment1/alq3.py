from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag 
from nltk.stem.snowball import SnowballStemmer  
from nltk.stem import WordNetLemmatizer
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import re

num=5331 #number of reviews in each text file

###########date prepration and preprocessing######################
pos=open('./rt-polaritydata/rt-polaritydata/rt-polarity.pos',encoding='latin-1')
neg=open('./rt-polaritydata/rt-polaritydata/rt-polarity.neg',encoding='latin-1')

pos_review=list()
neg_review=list()

stop_words = stopwords.words('english')
for w in ['!',',','.','?','-s','-ly','\'s']:
    stop_words.append(w)

def preprocessing(line,stop_words):

	stemmer=SnowballStemmer('english')

	word_tokens=word_tokenize(line)
	word_stemmed=[stemmer.stem(w) for w in word_tokens]
	word_filtered=[w for w in word_stemmed if not w in stop_words]
	word_sub=[re.sub('[0-9+]','',w) for w in word_filtered]
	line_clean=' '.join(word_sub)
	return line_clean


for line in pos:
	line_clean=preprocessing(line,stop_words)
	pos_review.append(line_clean)
for line in neg:
	line_clean=preprocessing(line,stop_words)
	pos_review.append(line_clean)

X=pos_review+neg_review
Y= ['positive' for n in range(num)]+['negative' for n in range(num)]

###############split train(80%), develop(10%) and test data(10%)####################
train_x, develop_test_x, train_y, develop_test_y = model_selection.train_test_split(X,Y,test_size=0.2,random_state=8)
develop_x, test_x, develop_y, test_y = model_selection.train_test_split(develop_test_x,develop_test_y,test_size=0.5,random_state=8)

###############feature extraction and vectorize####################
vect =CountVectorizer(min_df=2)
vect.fit(train_x)
#print("#feature=",len(vect.get_feature_names()))
Vtrain = vect.transform(train_x)
Vdevelop=vect.transform(develop_x)
Vtest=vect.transform(test_x)

#############Train the 3 classifiers using the training data, and tune hyperparameters on the development set
###############SVM classifier####################
clf_svm=SVC(C=2,gamma='scale')
clf_svm.fit(Vtrain,train_y)
svm_acc=clf_svm.score(Vdevelop,develop_y)
print ("SVM accuracy=", svm_acc)

###############Logistic Regression classifier####################
clf_lr = LogisticRegression(C=1.2,random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=2000)
clf_lr.fit(Vtrain, train_y)
lr_acc=clf_lr.score(Vdevelop,develop_y)
print("LogisticRegression accuracy=", lr_acc)

###############Naive Bayes classifier####################
clf_nb = MultinomialNB(alpha=1.0)
clf_nb.fit(Vtrain, train_y)
nb_acc=clf_nb.score(Vdevelop,develop_y)
print("NaiveBayes accuracy=", nb_acc)

###############Select NB and use it to do classification on the test set####################
final_acc=clf_nb.score(Vtest,test_y)
print("Final test accuracy using Naive Bayes model: ", final_acc)
confusion_matrix=[[0,0],[0,0]]
for i in range(len(test_y)):
	r=clf_nb.predict(Vtest[i])
	if(test_y[i]=='positive'):
		if(r=='positive'):
			confusion_matrix[0][0]+=1
		else:
			confusion_matrix[0][1]+=1
	else:
		if(r=='positive'):
			confusion_matrix[1][0]+=1
		else:
			confusion_matrix[1][1]+=1
print("Confusion Matrix=",confusion_matrix)
