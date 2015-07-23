##### Part 1 #####
import pandas as pd
import numpy as np
import seaborn as sns

# 1. read in the yelp dataset
yelp = pd.read_csv('hw/data/yelp.csv')
yelp.head()
yelp.describe()
yelp.shape
sns.pairplot(yelp)
yelp.corr()
sns.heatmap(yelp.corr())

# 2. Perform a linear regression using 
# "stars" as your response and 
# "cool", "useful", and "funny" as predictors
from sklearn.linear_model import LinearRegression

feature_cols = ['cool', 'useful', 'funny']
X = yelp[feature_cols]
y = yelp.stars

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print linreg.intercept_
print linreg.coef_
zip(feature_cols, linreg.coef_)

# Visualization
sns.pairplot(yelp, x_vars=['cool','useful','funny'], y_vars='stars', size=4.5, aspect=0.7, kind='reg')

# Linear Regression Model using STATS Model; [How do I use training data only for STATS?]
import statsmodels.formula.api as smf
lm = smf.ols(formula='stars ~ cool + useful + funny', data=yelp).fit()
lm.params

# 3. Show your MAE, R_Squared and RMSE
# R_Squared [Can I easily calculated R_Squared without STATS model?]
lm.rsquared

# MAE
from sklearn import metrics
y_pred = linreg.predict(X_test)
metrics.mean_absolute_error(y_test, y_pred)

# RMSE
np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# 4. Use statsmodels to show your pvalues
# for each of the three predictors
# Using a .05 confidence level, 
# Should we eliminate any of the three?
print lm.pvalues
lm.conf_int()
# [STATS model doesn't ]

# Keep all three features given statistically significant p-value

# 5. Create a new column called "good_rating"
# this could column should be True iff stars is 4 or 5
# and False iff stars is below 4
yelp['good_rating'] = yelp['stars'] >= 4

# 6. Perform a Logistic Regression using 
# "good_rating" as your response and the same
# three predictors
y_logreg = yelp.good_rating
X_train, X_test, y_logregtrain, y_logregtest = train_test_split(X, y_logreg, random_state=1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_logregtrain)
logreg.intercept_
zip(feature_cols, logreg.coef_[0])
y_logregpred = logreg.predict(X_test)

# 7. Show your Accuracy, Sensitivity, Specificity
# and Confusion Matrix
matrix = metrics.confusion_matrix(y_logregtest, y_logregpred)

sensitivity = float(matrix[1][1]) / (matrix[1][0] + matrix[1][1])
specificity = float(matrix[0][0]) / (matrix[0][1] + matrix[0][0])
accuracy = (float(matrix[0][0]) + matrix[1][1]) / ((matrix[1][0] + matrix[1][1])+(matrix[0][1] + matrix[0][0]))

# 8. Perform one NEW operation of your 
# choosing to try to boost your metrics!
yelp['cool^2'] = yelp['cool'] ** 2
yelp['useful^2'] = yelp['useful'] ** 2
yelp['funny^2'] = yelp['funny'] ** 2
yelp['cool*funny'] = yelp['funny'] * yelp['cool']
yelp['cool*useful'] = yelp['useful'] * yelp['cool']
yelp['cool+funny'] = yelp['funny'] + yelp['cool']

feature_cols_sqrd = ['cool', 'useful', 'funny', 'cool*funny']
X_sqrd = yelp[feature_cols_sqrd]

X_sqrdtrain, X_sqrdtest, y_logregtrain, y_logregtest = train_test_split(X_sqrd, y_logreg, random_state=1)

logreg_sqrd = LogisticRegression()
logreg_sqrd.fit(X_sqrdtrain, y_logregtrain)
logreg_sqrd.intercept_
zip(feature_cols, logreg_sqrd.coef_[0])
y_sqrd_logregpred = logreg_sqrd.predict(X_sqrdtest)

matrix = metrics.confusion_matrix(y_logregtest, y_sqrd_logregpred)

sensitivity = float(matrix[1][1]) / (matrix[1][0] + matrix[1][1])
specificity = float(matrix[0][0]) / (matrix[0][1] + matrix[0][0])
accuracy = (float(matrix[0][0]) + matrix[1][1]) / ((matrix[1][0] + matrix[1][1])+(matrix[0][1] + matrix[0][0]))

# [review count by business]
'''
data = []
for line in weather:
    data.append(line.split(','))
'''

##### Part 2 ######

# 1. Read in the titanic data set.
titanic = pd.read_csv('hw/data/titanic.csv')
titanic.info()
titanic.head()
titanic.describe()
titanic.shape
sns.pairplot(titanic) # [Error message]
titanic.corr()
sns.heatmap(titanic.corr())


# 4. Create a new column called "wife" that is True
# if the name of the person contains Mrs.
# AND their SibSp is at least 1

def wife_col(row):
    if 'Mrs.' in row.Name and row.SibSp >= 1:                
        return 1
    return 0

titanic['wife'] = titanic.apply(wife_col, axis=1)


# 5. What is the average age of a male and
# the average age of a female on board?
titanic.groupby('Sex').Age.mean()
titanic.groupby('Sex').agg([np.min, np.max])

# 5. Fill in missing MALE age values with the
# average age of the remaining MALE ages
titanic.Age.isnull().sum()

def adj_blank_age(row):
    if row.Sex == 'male' and row.Age > 0:
        return row.Age
    elif row.Sex == 'male':
        return 31
    elif row.Sex == 'female' and row.Age > 0:
        return row.Age
    elif row.Sex == 'female':
        return 28

titanic['Age_adj'] = titanic.apply(adj_blank_age, axis=1)
titanic.head(20) #check

titanic.Age_adj.isnull().sum()

# [how to code more efficiently]


# 6. Fill in missing FEMALE age values with the
# average age of the remaining FEMALE ages


# 7. Perform a Logistic Regression using
# Survived as your response and age, wife
# as predictors

from sklearn.cross_validation import train_test_split

titanic_features = ['Age_adj', 'wife']
X_titanic = titanic[titanic_features]
y_titanic = titanic.Survived

X_titanic_train, X_titanic_test, y_titanic_train, y_titanic_test = train_test_split(X_titanic, y_titanic, random_state=1)

from sklearn.linear_model import LogisticRegression
logreg_titanic = LogisticRegression()
logreg_titanic.fit(X_titanic_train, y_titanic_train)
logreg_titanic.intercept_

zip(titanic_features, logreg_titanic.coef_[0])
y_titanic_pred = logreg_titanic.predict(X_titanic_test)

# 8. Show Accuracy, Sensitivity, Specificity and 
# Confusion matrix

from sklearn import metrics
matrix = metrics.confusion_matrix(y_titanic_test, y_titanic_pred)

sensitivity = float(matrix[1][1]) / (matrix[1][0] + matrix[1][1])
specificity = float(matrix[0][0]) / (matrix[0][1] + matrix[0][0])
accuracy = (float(matrix[0][0]) + matrix[1][1]) / ((matrix[1][0] + matrix[1][1])+(matrix[0][1] + matrix[0][0]))


# 9. now use ANY of your variables as predictors
# Still using survived as a response to boost metrics!

from sklearn.cross_validation import train_test_split

titanic_features2 = ['Age_adj', 'wife','Fare','Pclass']
X_titanic2 = titanic[titanic_features2]
y_titanic2 = titanic.Survived

X_titanic2_train, X_titanic2_test, y_titanic2_train, y_titanic2_test = train_test_split(X_titanic2, y_titanic2, random_state=1)

from sklearn.linear_model import LogisticRegression
logreg_titanic2 = LogisticRegression()
logreg_titanic2.fit(X_titanic2_train, y_titanic2_train)
logreg_titanic2.intercept_

zip(titanic_features2, logreg_titanic2.coef_[0])
y_titanic2_pred = logreg_titanic2.predict(X_titanic2_test)

# 10. Show Accuracy, Sensitivity, Specificity
from sklearn import metrics
matrix = metrics.confusion_matrix(y_titanic2_test, y_titanic2_pred)

sensitivity = float(matrix[1][1]) / (matrix[1][0] + matrix[1][1])
specificity = float(matrix[0][0]) / (matrix[0][1] + matrix[0][0])
accuracy = (float(matrix[0][0]) + matrix[1][1]) / ((matrix[1][0] + matrix[1][1])+(matrix[0][1] + matrix[0][0]))


# REMEMBER TO USE
# TRAIN TEST SPLIT AND CROSS VALIDATION
# FOR ALL METRIC EVALUATION!!!!

