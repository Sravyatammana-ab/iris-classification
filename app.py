
from flask import Flask,render_template,request
from sklearn.datasets import load_iris
import numpy as np
import model
import pickle as pkl
app=Flask(__name__,template_folder="templates")
model=pkl.load(open('model.pkl','rb'))
@app.route('/')
def welcome():
    return render_template('iris_p.html')

@app.route('/predict',methods=['GET'])
def predict():
    
    sepallength=float(request.args.get("sepallength"))
    sepalwidth=float(request.args.get("sepalwidth"))
    petallength=float(request.args.get("petallength"))
    petalwidth=float(request.args.get("petalwidth"))
    arr=np.array([[sepallength,sepalwidth,petallength,petalwidth]],dtype=float)
    iris=load_iris()
    prediction=model.predict(arr)
    out=iris.target_names[prediction][0]
    return render_template('out.html', output=out)
if __name__ == "__main__":
    app.run(debug=True)