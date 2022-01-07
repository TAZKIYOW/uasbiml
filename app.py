import numpy as np
from flask import Flask, render_template, request
import pickle
	

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
	

@app.route('/')
def home():
    return render_template('index.html')
	

@app.route('/predict',methods=['POST'])
def predict():
	    '''
	    For rendering results on HTML GUI
	    '''
	    int_features = [int(x) for x in request.form.values()]
	    final_features = [np.array(int_features)]
	    prediction = model.predict(final_features)
	

	    output = round(prediction[0], 2)
	

	    return render_template('index.html', prediction_text='Prediksi Harga Tanah {}'.format(output))
if __name__ == "__main__":
	    app.run(debug=True)


from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__, template_folder='templates')
@app.route('/')
def student():
   return render_template("home.html")
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1,1)
    loaded_model = joblib.load('model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]
@app.route('/',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
    to_predict_list = request.form.to_dict()
    to_predict_list=list(to_predict_list.values())
    to_predict_list = list(map(float, to_predict_list))
    result = float(ValuePredictor(to_predict_list))
    return render_template("home.html",result = result)
if __name__ == '__main__':
   app.run(debug = True)