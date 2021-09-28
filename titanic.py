#Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Load the dataset
titanic = sns.load_dataset('titanic')

#------------------------------------------------------------------------------------------------------
#print the first 10 rows of the data
#print(titanic.head(10))
 
#------------------------------------------------------------------------------------------------------
#Count the number  of rows and columnsin in the dataset
#print(titanic.shape)

#------------------------------------------------------------------------------------------------------
#Get some stadistics
#print(titanic.describe())  #We need to coment this at the end

#------------------------------------------------------------------------------------------------------
#Get the number of survivors
#print(titanic['survived'].value_counts())

#------------------------------------------------------------------------------------------------------
#Visualize the count of survivors 
sns.countplot( titanic['survived'] )                                                   
plt.show()

#------------------------------------------------------------------------------------------------------
#Visualize the count of survivors for columns  'who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked'
# cols = ['who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked'] 
# n_rows = 2
# n_cols = 3

#------------------------------------------------------------------------------------------------------
#The subplot grid an figure size of each graph                                          #it gave me an error
# fig, axs = plt.subplots(n_rows,n_cols)
# for r in range(0,n_rows):
#     for c in range(0,n_cols):
#         i = r*n_cols + c #index to go trought the number of columns 
#         ax = axs[r][c] #show where to position each subplot
#         sns.countplot(titanic[cols[i]], hue=titanic['survived'], ax=ax)
#         ax.set_title(cols[i])
#         ax.legend(title = 'survived', loc = 'upper right')

# plt.tight_layout()
# plt.show()


#------------------------------------------------------------------------------------------------------
#look at survival rate by sex                                                           it works
#print(titanic.groupby('sex')[['survived']].mean())

#------------------------------------------------------------------------------------------------------
#look at survival rates by sex an class                                                     
#print(titanic.pivot_table('survived', index = 'sex', columns = 'class'))

#------------------------------------------------------------------------------------------------------
#look at survival rates by sex an class visually                                         #it gave me an error
# titanic.pivot_table('survived', index = 'sex', columns = 'class').plot()
# plt.show()

#------------------------------------------------------------------------------------------------------
#Plot the survival rate of each class                                                       it works
# sns.barplot(x='class', y='survived', data= titanic)
# plt.show()

#------------------------------------------------------------------------------------------------------
#Look at the survival rate by sex, age and class                                          it works
# age = pd.cut(titanic['age'], [0,18,80] )
# print(titanic.pivot_table('survived', ['sex', age], 'class' ))
#------------------------------------------------------------------------------------------------------

#Plot the prices paid of each class
# plt.scatter(titanic['fare'], titanic['class'], color =  'purple', label = 'Passenger Paid' )
# plt.ylabel('Class')
# plt.xlabel('Price / Fare')
# plt.title('Price of each Class')
# plt.legend()
# plt.show()
#------------------------------------------------------------------------------------------------------

#Coutn the empty valuesin each columns
#print(titanic.isna().sum())
#------------------------------------------------------------------------------------------------------

#look at all of the values in each column & get a count
# for val in titanic:
#     print(titanic[val].value_counts())
#     print()
#------------------------------------------------------------------------------------------------------

#Drop the columns 
titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'alone', 'adult_male'], axis=1)

# #Remove the rows with missing values
titanic = titanic.dropna(subset = ['embarked', 'age'])
#------------------------------------------------------------------------------------------------------

#Count the new number of rows and columns  in the data set
print(titanic.shape)
#------------------------------------------------------------------------------------------------------

#look at the data types
#print(titanic.dtypes)
#------------------------------------------------------------------------------------------------------

#Encode the sex column
titanic.iloc[:, 2 ] = labelencoder.fit_transform( titanic.iloc[:, 2 ]. values )
#Encode the embarked column
titanic.iloc[:, 7 ] = labelencoder.fit_transform( titanic.iloc[:, 7 ]. values )
#------------------------------------------------------------------------------------------------------

#print(titanic.dtypes)
#------------------------------------------------------------------------------------------------------

#Split the data into independant X and dependant Y variables
X = titanic.iloc[:, 1:8].values
Y = titanic.iloc[:, 0].values
#------------------------------------------------------------------------------------------------------

#Split the dataset in 80% training and 20% testing 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
#------------------------------------------------------------------------------------------------------

#Scale the data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#------------------------------------------------------------------------------------------------------

#Create a function with many machine learnings models
def models(X_train, Y_train):
#------------------------------------------------------------------------------------------------------

    #Use logistic regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, Y_train)
    #------------------------------------------------------------------------------------------------------

    #Use KNeighbors
    from sklearn.neighbors import KNeighborsClassifier
    knn= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
    knn.fit(X_train, Y_train)
    #------------------------------------------------------------------------------------------------------

    #Use SVC (linear kernel)
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear', random_state= 0)
    svc_lin.fit(X_train,Y_train)
    #------------------------------------------------------------------------------------------------------

    #Use    SVC (RBF kernel)
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train,Y_train)
    #------------------------------------------------------------------------------------------------------

    #Use GaussianNB
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)
    #------------------------------------------------------------------------------------------------------

    #Use a decision tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train,Y_train)
    #------------------------------------------------------------------------------------------------------

    #Use the ramdon forest
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train,Y_train)
    #------------------------------------------------------------------------------------------------------

    #print the training accuracy for each model
    print('[0] Logistic Regression Training Accuracy', log.score(X_train,Y_train))
    print('[1] KNeighbors Training Accuracy', knn.score(X_train,Y_train))
    print('[2] SVC Linear Training Accuracy', svc_lin.score(X_train,Y_train))
    print('[3] SVC RBF Training Accuracy', svc_rbf.score(X_train,Y_train))
    print('[4] Gaussian NB Training Accuracy', gauss.score(X_train,Y_train))
    print('[5] Decision Tree Training Accuracy', tree.score(X_train,Y_train))
    print('[6] Random Forest Training Accuracy', forest.score(X_train,Y_train))

    #------------------------------------------------------------------------------------------------------
    return log, knn, svc_lin, svc_rbf, gauss, tree, forest
#------------------------------------------------------------------------------------------------------

#Get and train all the models
model = models(X_train,Y_train)
#------------------------------------------------------------------------------------------------------

#Show the confusion matrix and accuracy for all of the models on the test data
from sklearn.metrics import confusion_matrix

for i in range ( len(model)):
    cm = confusion_matrix(Y_test, model[i].predict(X_test))

#------------------------------------------------------------------------------------------------------

#Extract TN, Fp, FN; TP
TN, FP, FN, TP = confusion_matrix(Y_test, model[i].predict(X_test)).ravel()

test_score = (TP+TN) / (TP+TN+FN+FP)

print(cm)
print('Model [{}] Testing Accuracy = "{}"'.format( i, test_score))
print()

# Get  features importance
forest = model[6]
importances = pd.DataFrame({'feature': titanic.iloc[:, 1:8].columns,  'importance': np.round(forest.feature_importances_, 3)})
importances = importances.sort_values('importance',ascending = False).set_index('feature')
print(importances)
#------------------------------------------------------------------------------------------------------

#Visualize importances
importances.plot.bar()
plt.show()
#------------------------------------------------------------------------------------------------------

#print the prediction of the random forest
pred = model[6].predict
print(pred)

print()

#Print actual values
print(Y_test)
#------------------------------------------------------------------------------------------------------
#My survival
my_survival = [[1,0,21,0,0,0,1]]

#Scaling my survival
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
my_survival_scaled = sc.fit_transform(my_survival_scaled)

#Print prediction of my survival using random forest classifier
pred = model[6].predict(my_survival_scaled)
print(pred)

if pred == 0:
    print('you did not make it')
else:
    print('Ypu survived')

#------------------------------------------------------------------------------------------------------