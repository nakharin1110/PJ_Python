##https://www.youtube.com/watch?v=lN8AsMRiKP4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv('C:\\Users\\NAKHARIN\\Documents\\GitHub\\PJ_Python\\__MACOSX\\student_scores.csv')

dataset.shape
(25,2)

dataset.head()
dataset.describe()

dataset.plot(x='Hours',y='Scores',style="*")
plt.title('Student prediction')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.show()

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,1].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

regressor=LinearRegression()
regressor.fit(X_train,Y_train)

print(regressor.intercept_)

print(regressor.coef_)

y_pred=regressor.predict(X_test)
df=pd.DataFrame({'Actual':Y_test,'Predicted':y_pred})
