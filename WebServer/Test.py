# flask for web app.
import flask as fl
from flask import request, jsonify 
# numpy for numerical work.
import numpy as np
import random

# Create a new web app.
app = fl.Flask(__name__)

# Add root route.
@app.route("/")
def home():
    #return "Hello, World!"
    return app.send_static_file('index.html')

# Add normal route.
@app.route('/api/windspeed')
def windspeed():
    #value = request.form['input']
    #value = random.randint(0, 25)
    value = request.form["input"]
    return{"value":value} 
