from flask import Flask, render_template, request
from dummies_dic.dummies import Suger_dummies , car_type , destination_dummies , icon_dummies , name_dummies,product_id_dummies ,source_dummies
from dummies_dic.dummies import Uv_dummies , Month_dummies
import joblib

app = Flask(__name__)

# import the model and his scaler
scaler = joblib.load('Machine learning Model/scaler.h5')
model = joblib.load('Machine learning Model/model.h5')


@app.route('/')
def index():
    return render_template('index.html')




@app.route('/home')
def home():
    all_data = request.args
    name = name_dummies[all_data["name"]]
    car_tyb = car_type[all_data['Car type']]
    dest = destination_dummies[all_data['Destination']]
    source = source_dummies[all_data['source']]
    product = product_id_dummies[all_data['Product ID']]
    suger = Suger_dummies[all_data['Suger']]
    UV = Uv_dummies[all_data['UV']]
    Month = Month_dummies[all_data['Month']]
    Icon = icon_dummies[all_data['Icon']]
    Distance = float(all_data['Distance'])
    data =[Month, suger , UV , Distance  ] + source + dest + product + name + Icon + car_tyb
    data = scaler.transform([ data])
    pred = round(model.predict(data)[0])    
    return render_template('prediction.html' , pred = pred)



if __name__ == "__main__":
    app.run('127.0.0.1', port = 5500)
    
'''
 all_data = request.args
    name = name_dummies[all_data["name"]]
    car_tyb = car_type[all_data['Car type']]
    dest = destination_dummies[all_data['Destination']]
    source = source_dummies[all_data['source']]
    product = product_id_dummies[all_data['Product ID']]
    suger = Suger_dummies[all_data['Suger']]
    UV = Uv_dummies[all_data['UV']]
    Month = Month_dummies[all_data['Month']]
    Icon = icon_dummies[all_data['Icon']]
    Distance = float(all_data['Distance'])
    data = Month + suger + UV + [Distance] + source + dest + product + name + Icon + car_tyb 
    data = scaler.transform([data])
    pred = round(model.predict(data)[0])
'''