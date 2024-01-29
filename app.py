from flask import Flask,render_template,request
import pickle

cv=pickle.load(open("models/cv.pkl","rb"))
model=pickle.load(open("models/clf.pkl","rb"))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html") # as the index.html is in templates folder so no need to give path as templates/index.html

@app.route("/predict",methods=["POST"])
def predict():
    email_text=request.form.get('email-content')
    tokenized_email=cv.transform([email_text])
    predictions=model.predict(tokenized_email)
    predictions=1 if predictions==1 else -1
    return render_template("index.html",predictions=predictions,email_text=email_text)

if __name__=="__main__":
    app.run(debug=True)   #debuge=True will refresh web after the change made in code

