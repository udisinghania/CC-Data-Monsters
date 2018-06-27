from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

a = load_iris()
d = pd.DataFrame(data = np.c_[a['data'], a['target']], columns = a['feature_names']+['target'])

#print(d)

x = d[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)' ]];
# x is our singular data item for KNeighbors classification
#print(type(x))

y = d.target
#print(y)
# y is our target variable for KNeighbors classification
#print(type(y))

knn= KNeighborsClassifier(n_neighbors=5)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state = 42)

#print(x_train)
#print(y_train)
#our training and testing have been set now.


knn.fit(x_train, y_train)


y_pred = (knn.predict(x_test))
print(y_pred)

print(knn.score(x_test,y_test))

error = mean_squared_error(y_test, y_pred)
print(error)
  #the above code is a KNeighbors classification model that analyses an iris dataset loaded from the sklearn library


plt.scatter(x['sepal length (cm)'], y,color='red')
plt.scatter(x_test['sepal length (cm)'], y_pred, color = 'blue')
plt.show()

plt.scatter(x['sepal width (cm)'], y, color='red')
plt.scatter(x_test['sepal width (cm)'], y_pred,  color = 'blue')
plt.show()

plt.scatter(x['petal length (cm)'], y, color='red')
plt.scatter(x_test['petal length (cm)'], y_pred,  color = 'blue')
plt.show()

plt.scatter(x['petal width (cm)'], y, color='red')
plt.scatter(x_test['petal width (cm)'], y_pred,  color = 'blue')
plt.show()
