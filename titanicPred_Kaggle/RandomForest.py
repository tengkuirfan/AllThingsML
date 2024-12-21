import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

train = pd.read_csv('datasets/train_data.csv')
test = pd.read_csv('datasets/test_data.csv')
test_survival = pd.read_csv('datasets/test_survival.csv')

train.drop(columns=['Cabin'], inplace=True)
test.drop(columns=['Cabin'], inplace=True)

imputer = KNNImputer(n_neighbors=5)
train[['Age']] = imputer.fit_transform(train[['Age']])
test[['Age']] = imputer.transform(test[['Age']])
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

train["Sex"] = train["Sex"].astype('category')
test["Sex"] = test["Sex"].astype('category')
train["Embarked"] = train["Embarked"].astype('category')
test["Embarked"] = test["Embarked"].astype('category')

test = test.merge(test_survival, on='PassengerId')
X_train = train.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket'])
X_test = test.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket'])
y_train = train['Survived']
y_test = test['Survived']

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Classification Report\n", classification_report(y_test, predictions))
print("Confusion Matrix\n", confusion_matrix(y_test, predictions))
print("Accuracy\n", accuracy_score(y_test, predictions))
