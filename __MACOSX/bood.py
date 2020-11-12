##https://www.youtube.com/watch?v=lN8AsMRiKP4

#Pandas เป็น Library ใน Python ที่ทำให้เราเล่นกับข้อมูลได้ง่ายขึ้น
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

## นำเข้าข้อมูล คือคะแนนนักเรียน
dataset=pd.read_csv('C:\\Users\\NAKHARIN\\Documents\\GitHub\\PJ_Python\\__MACOSX\\student_scores.csv')

dataset.shape
(25,2)

# คำอธิบายและเช็ค
dataset.head()
dataset.describe()

# ออกแบบกราฟ
dataset.plot(x='Hours',y='Scores',style="*")
plt.title('Student prediction')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.show()

# เลือกข้อมูล
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,1].values

# แบ่งข้อมูลสำหรับสอน (train) ออกเป็น 80% และสำหรับการทดสอบ (test) ออกเป็น 20% และกำหนด random_state=0 คือสุ่มทีละ 0 ตัว 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# วิธีการทางสถิติอย่างหนึ่งในการหาความสัมพันธ์ระหว่าง ตัวแปรต้น และ ตัวแปรตาม หาความสัมพันธ์ระหว่างชั่วโมงที่เรียนกับคะเเนนที่ได้
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

# แสดงค่าจุดตัดแกน y หรือค่า  α
print(regressor.intercept_)

# แสดงค่าสัมประสิทธิ์ Coefficient หรือค่า  β
print(regressor.coef_)

# ทำนายค่าที่คำนวณจากบรรทัด 40-43
y_pred=regressor.predict(X_test)
df=pd.DataFrame({'Actual':Y_test,'Predicted':y_pred})