from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# 1. Load Datasets
description = pd.read_csv("datasets/description.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
medication = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")
workout = pd.read_csv("datasets/workout_df.csv")
X = pd.read_csv("datasets/Training.csv").drop('prognosis', axis=1)
symptoms_dict = {symptom: index for index, symptom in enumerate(X.columns)}
Y = pd.read_csv("datasets/training.csv")['prognosis']
diseases_list = sorted(list(Y.unique()))
# 2. Load Model
svc = pickle.load(open(r"C:\Users\Ananya\Documents\models\svc.pkl", "rb"))

# --- Paste your get_predicted_value and helper functions here ---
#model prediction 
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = "".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre = [col for col in pre.values]

    med = medication[medication['Disease'] == dis]['Medication']
    med = [m for m in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [d for d in die.values]

    # Change the variable on the left to something like 'wrk' or 'wrkout'
    wrk = workout[workout['disease'] == dis]['workout'] 
    wrkout_list = [w for w in wrk.values] # Use the new variable here too

    return desc, pre, med, die, wrkout_list # Return the new list name
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict: # Good to add this check for safety
            input_vector[symptoms_dict[item]] = 1
            
    # Convert to DataFrame to match training format and stop the warning
    input_df = pd.DataFrame([input_vector], columns=symptoms_dict.keys())
    return diseases_list[svc.predict(input_df)[0]]
        
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    # 1. Initialize variables so they exist for the 'return' line
    predicted_disease = ""
    desc = ""
    pre = []
    med = []
    die = []
    wrkout = []

    if request.method == 'POST':
        # 2. Get symptoms from the form
        symptoms = request.form.get('symptoms')
        
        if symptoms:
            # 3. Clean and process (use your notebook logic here)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [sym.strip("[] '") for sym in user_symptoms]
            
            # 4. Call your functions to fill the variables
            predicted_disease = get_predicted_value(user_symptoms)
            desc, pre, med, die, wrkout = helper(predicted_disease)

    # 5. Return the template (Variables now exist no matter what)
    return render_template('index.html', 
                           predicted_disease=predicted_disease, 
                           dis_desc=desc, 
                           dis_pre=pre, 
                           dis_med=med, 
                           dis_die=die, 
                           dis_wrkout=wrkout)
if __name__ == "__main__":
    app.run(debug=True)