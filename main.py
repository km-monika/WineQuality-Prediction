import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
import seaborn as sns

data=pd.read_csv('winequality-red.csv')

reviews=[]
for i in data['quality']:
    if i>=1 and i<=3:
        reviews.append('1')
    elif i>=4 and i<=6:
        reviews.append('2')
    elif i>=7 and i<=10:
        reviews.append('3')
data['reviews']=reviews

x=data[['fixed acidity', 'volatile acidity','citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
y=data['reviews']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

from sklearn.ensemble import RandomForestClassifier
rt_model=RandomForestClassifier(random_state=1,n_estimators=100)
rt_model.fit(x_train,y_train)

y_pred=rt_model.predict(x_test)
print("Accuracy score is ",accuracy_score(y_test,y_pred))
    

from flask import Flask,render_template,request
app=Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/", methods=['POST'])
def getvalue():
    fixed=request.form["fixed_acidity"]
    volatile=request.form["volatile_acidity"]
    citric=request.form["citric_acid"]
    residual=request.form["residual_sugar"]
    chlorides=request.form["chlorides"]
    free=request.form["free_sulfur_dioxide"]
    total=request.form["total_sulfur_dioxide"]
    density=request.form["density"]
    ph=request.form["pH"]
    sulphates=request.form["sulphates"]
    alcohol=request.form["alcohol"]
    z=(fixed,volatile,citric,residual,chlorides,free,total,density,ph,sulphates,alcohol)
    print(z)
    import numpy as np
    import pandas as pd
    data=np.array([z])
    x=pd.DataFrame(data=data)
    y=rt_model.predict(x)
    if y=='1':
        print("Wine quality is Bad")
        w="Wine quality is BAD with a score ranging from 0 to 3 out of 10"
    elif y=='2':
        print("Wine quality is Good")
        w="Wine quality is AVERAGE ranging from a score of 4 to 6 out of 10"
    elif y=='3':
        print("Wine quality is Very Good")
        w="Wine quality is Good with a score ranging between 7 to 10 out of 10"
        
    return render_template("pass.html",z=w)

if __name__=="__main__":
    app.run()