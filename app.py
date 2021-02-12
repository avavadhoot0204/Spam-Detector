from flask import Flask,render_template,url_for,request, abort, session,redirect
from flask_mail import Mail, Message
import pickle
import joblib


filename = 'pickle.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)




mail=Mail(app)

@app.route('/')
def home():
	return render_template('home_new_new.html')
		

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True, use_reloader=False)