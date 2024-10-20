from flask import Flask,render_template,request
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def prediction():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
                Company=request.form.get('company'),
                TypeName=request.form.get('typeName'),
                Ram=int(request.form.get('ram')) if request.form.get('ram') is not None else 0,  # Default value if None
                Gpu=request.form.get('gpu'),
                Touchscreen=1 if request.form.get('touchscreen') == 'Yes' else 0,  # Convert Yes/No to int
                IPS_screen=1 if request.form.get('IPS_screen') == 'Yes' else 0,  # Convert Yes/No to int
                CPU_Brand=request.form.get('processor'),
                Weight=float(request.form.get('weight')) if request.form.get('weight') is not None else 0.0,  # Default if None
                HDD=int(request.form.get('HDD')) if request.form.get('HDD') is not None else 0,  # Default if None
                SSD=int(request.form.get('SDD')) if request.form.get('SDD') is not None else 0,  # Default if None
                OS=request.form.get('OS'),
                PPI=float(request.form.get('ppi')) if request.form.get('ppi') is not None else 0.0  # Default if None

                            )
        pred_df = data.get_data_as_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print(results)
        return render_template('home.html',results=results[0])

if __name__ == "__main__":
    app.run(host = "0.0.0.0",debug=True)
