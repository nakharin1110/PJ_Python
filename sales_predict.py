import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt

def regression_model(model, df1, df2, name):
  x = df1[['month','year','laptop','mobile']]
  y = df1[['revenue']]
  model.fit(x,y)##ใช้เมธอด fit เพื่อนำข้อมูลมาหาค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน
  x_test = df2[['month','year','laptop','mobile']]
  y_test = df2[['revenue']]
  predictions = model.predict(x_test)
  accuracy = model.score(x_test,y_test)
  print ('Accuracy : %s' % '{0:.5%}'.format(accuracy), name)
  newarray = x_test[['month']].values
  newarray2 = newarray.ravel()
  newarray3 = x_test[['year']].values
  newarray4 = newarray3.ravel()
  newarray5 = predictions.ravel()
  newdf = pd.DataFrame({'revenue':newarray5, 'month':newarray2, 'year':newarray4})
  sns.factorplot(x="month", y="revenue", hue="year", data=newdf)
  plt.show()

data = pd.read_csv("C:\\Users\\NAKHARIN\\Documents\\GitHub\\PJ_Python\\testing.csv")
df = pd.DataFrame(data)
x = df[['month','year','laptop','mobile']]
y = df[['revenue']]
sns.factorplot(x="month", y="revenue", hue="year", data=df)
plt.show()


