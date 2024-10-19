from flask import Flask,render_template,request
import numpy as np
import pandas as pd

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',method=['GET','POST'])
def prediction():
    if request.method == "GET":
        return render_template('home.html')
    else:
        pass
    
