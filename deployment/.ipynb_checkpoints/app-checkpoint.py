from flask import Flask
from flask import request
from flask import render_template
import time

import pickle
model = pickle.load(open("../deployment/models/rforest_hyp.pkl", "rb"))

from process import run_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("/index.html")

@app.route('/', methods=['POST'])
def form():
    customer_id = request.form['customer_id']
    no_data, df = run_model(model, customer_id)
    if no_data:
        return render_template("no_data.html", display=customer_id)
    else:
        return render_template("main.html", tables=[df.to_html(classes='data')])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)