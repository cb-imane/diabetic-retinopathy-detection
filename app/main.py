import pickle
import requests
from flask import Flask, render_template, request
from pydantic import BaseModel, Field, PositiveFloat


DR_CLASSES = {
    0: 'No Diabetic Retinopathy',
    1: 'Positive Retinopathy'

}
app = Flask(__name__)
#model = pickle.load(open(f"../src/trained_models/1f51d988d0ac47e9b6faeb95a851a27b.pkl", "rb"))
#model = pickle.load(f"../src/trained_models/e2a94cc00a284a2a865804926769e21f.pkl","rb")
model=pickle.load(open("../src/trained_models/65beb5dc65504ab490177d5b45138e0b.sav","rb"))


class FormQuery(BaseModel):
    #train_id: str
    ma1: int
    exudate1: float
    exudate2: float
    exudate3: float
    exudate31: float
    exudate5: float
    macula_opticdisc_distance: float
    opticdisc_diameter: float
    



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/diagnosis/", methods=["GET"])
def dr_features():
    return render_template("form.html")

@app.route("/prediction/", methods=["GET","POST"])
def get_prediction():
    if request.method == 'POST':
        ma1 = request.form.get('Ma1',22)

        exudate1 = request.form.get('Exudate1',49.895756)
        exudate2 = request.form.get('Exudate2',17.775994)
        exudate3 = request.form.get('Exudate3',5.270920)
        exudate31 = request.form.get('Exudate31',0.771761)
        exudate5 = request.form.get('Exudate5',0.018632	)
        macula_opticdisc_distance = request.form.get('MaculaDistance',0.486903)
        opticdisc_diameter = request.form.get('OpticdiscDiameter',0.100025)
        
        pred = model.predict(
            [
                [
                    ma1,
                    exudate1,
                    exudate2,
                    exudate3,
                    exudate31,
                    exudate5,
                    macula_opticdisc_distance,
                    opticdisc_diameter
                    
                ]
            ]
        )[0]
        print("working",pred)
    return render_template("prediction.html",prediction=DR_CLASSES[pred])




@app.route("/predict_from_api/", methods=["POST"])
def api_result():
    model_list = requests.get("http://127.0.0.1:8000/model/list/").json()
    if len(model_list) == 0:
        raise Exception("No model could be retrieved from the model registry")

    best_model = sorted(model_list, key=lambda d: d["recall_score"])[0]
    app.logger.debug(f"Best model retrieved : {best_model}")

    api_response = requests.post(
        "http://127.0.0.1:8000/model/predict/",
        json={
            **{"train_id": best_model["train_id"]},
            **FormQuery(**request.form.to_dict(flat=True)).model_dump(),
        },
    )

    response = api_response.json()
    app.logger.debug(response)

    return render_template("diagnosis.html", prediction=response["result"])



if __name__ == "__main__":
    
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True)
