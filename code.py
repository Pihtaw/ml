import pandas as pd 
import numpy as np
import statistics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import math

data = pd.read_csv('data.csv',  sep=';', header = 0, encoding='ISO-8859-1')
data = pd.DataFrame(data)

le = LabelEncoder()
columns = ['main_category', 'currency',  'state',  'country']
for i in columns:
  data[i] = le.fit_transform(data[i]) #дополняет пустые значения
  
columns1=['main_category', 'currency', 'goal', 'pledged', 'state', 'backers', 'country', '2 pledged', '2_pledged_real', '2_goal_real']
for i in columns1:
  data[i] /= data[i].std() 
  data[i] -= data[i].mean()
  
train, test = train_test_split(data, test_size = 0.5)

y_train = train
y_train = y_train.drop(columns=['main_category', 'currency', 'goal', 'pledged', 'backers', 'country', '2 pledged', '2_pledged_real', '2_goal_real']) 
print(y_train)
print('#' * 100)
x_train = train.drop(columns=['state']) 
print(x_train)


y_test = test
y_test = y_test.drop(columns=['main_category', 'currency', 'goal', 'pledged', 'backers', 'country', '2 pledged', '2_pledged_real', '2_goal_real']) 
print(y_test)
print('#' * 100)
x_test = test.drop(columns=['state']) 
print(x_test)

def func(a1, a2):
  r = 0
  for i in range(len(a1)):
    if(a1[i] == a2[i]):
      r = r + 1
  return r / len(a1)
a1 = [1, 2, 3, 4, 5]
a2 = [1, 2, 7, 4, 5]
print(func(a1, a2))

class KNearestNeighbors:
  def __init__(self, k):
    # сохраните в классе значение параметра k
    self.k = k
    
  def fit(self, X, y):
    # сохраните в классе обучающую выборку, чтобы находить ближайших соседей в ней
    self.X = np.asarray(X)
    self.y = np.asarray(y)
    
  def predict(self, X):
    # для данных объектов вернуть массив такой длины, как их количество, в котором будут предсказанные классы
    x_test = np.asarray(X)
    x_train = self.X
    y_train = self.y
    array = []
    ans = []
    a = []
    for i in range(x_test.shape[0]):
      massiv = pd.DataFrame(columns = {'dist', 'state'})
      for j in range(x_test.shape[0]):
        dist = math.sqrt((x_test[i][0] - x_train[j][0])**2+(x_test[i][1] - x_train[j][1])**2+(x_test[i][2] - x_train[j][2])**2+(x_test[i][3] - x_train[j][3])**2+(x_test[i][4] - x_train[j][4])**2+(x_test[i][5] - x_train[j][5])**2+(x_test[i][6] - x_train[j][6])**2+(x_test[i][7] - x_train[j][7])**2+(x_test[i][8] - x_train[j][8])**2)
        row = {'dist':dist, 'state':y_train[j]}
        massiv = massiv.append(row, ignore_index=True)
      array.append(massiv)
      summ = 0
      for i1 in range(len(array)):
        b = array[i1].nsmallest(self.k, 'dist')
        b = list(b['state'])
      if(sum(b) >= self.k // 2):
        a.append(1)
      else:
        a.append(0)
    return a
    
A = KNearestNeighbors(100)
A.fit(x_train, y_train)
answer = A.predict(x_test)

def func(a1, a2):
  r = 0
  for i in range(len(a1)):
    if(a1[i] == a2[i]):
      r = r + 1
  return r / len(a1)
  
answer2 = []
[answer2.append(int(i)) for i in y_train.values]
print(func(answer, answer2))
