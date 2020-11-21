# flask for web app.
import flask as fl
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

# Add uniform route.
@app.route('/api/uniform')
def uniform():
    return{"value":np.random.uniform()}

# Add normal route.
@app.route('/api/normal')
def normal():
    return{"value":np.random.normal()}


# Add normal route.
@app.route('/api/windspeed')
def windspeed():
    value = random.randint(0, 25)
    return{"value":value}