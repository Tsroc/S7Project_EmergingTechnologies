import flask as fl
from flask import request, jsonify 
from TrainedModel import PolynomialRegression

app = fl.Flask(__name__)

# Create the model
model = PolynomialRegression(2500, 3)
model.LoadData("Windspeed.txt")
model.CreateModel()


@app.route("/")
def home():
    return app.send_static_file('index.html')

@app.route('/api/windspeed', methods=["GET", "POST"])
def windspeed():
    if request.method == "POST":  
        value = request.form.get('input')
        prediction = model.PredictPower(int(value))
        print("The prediction is", prediction)
        return{"prediction":int(prediction)} 
    return app.send_static_file('index.html')
    