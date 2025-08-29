import sys
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


df= pd.read_csv('star.csv')
#color
st={'yellow':0,'blue':1,'white':2,'orange':3,'red':4,'white-yellow':5,'white-blue':6}
df['color']=df['color'].map(st)


#type
ar={'G':00,'O':10,'A':20,'K':30,'M':40,'F':50,'B':60}
df['type']=df['type'].map(ar)


print(df)


features=['color','luminosity','distance']
x=df[features]
y=df['type']

dtree = DecisionTreeRegressor()
dtree = dtree.fit(x, y)

plt.figure(figsize=(12,8))
tree.plot_tree(dtree, feature_names=features, filled=True)
plt.savefig("star tree.png")

sys.stdout.flush()


