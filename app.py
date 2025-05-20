from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)
app.secret_key = "your_secret_key"   
with open("construction_materials_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("budget_mapping.pkl", "rb") as f:
    budget_mapping = pickle.load(f)
with open("construction_type_encoder.pkl", "rb") as f:
    label_enc_type = pickle.load(f)
with open("materials_encoder.pkl", "rb") as f:
    label_enc_materials = pickle.load(f)

EXCEL_PATH = "construction_data_with_common_materials.xlsx"
df_excel = pd.read_excel(EXCEL_PATH)

phase_cols = [
    "Earthwork",
    "Structural Work",
    "Architectural Finishes",
    "Building Services",
    "Site Development",
    "Specialized Works"
]

# --- NEW: Database check function ---
def check_material_in_database(material_name):
    """Check if a material is present in the materials table."""
    conn = sqlite3.connect('materials.db')
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM materials WHERE LOWER(name) = LOWER(?)", (material_name,))
    count = cur.fetchone()[0]
    conn.close()
    return count > 0

@app.route('/')
def home():
    construction_types = [
        "Single-Family Homes",
        "Schools & Universities",
        "Hotels & Resorts",
        "Industrial Buildings",
        "Stadiums & Arenas",
        "Religious Buildings",
        "Apartments",
        "Offices",
        "Retail Stores & Malls",
        "Hospitals"
    ]
    budgets = ["Low", "Medium", "High"]
    return render_template(
        "index.html",
        construction_types=construction_types,
        budgets=budgets
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        budget = data['budget']
        construction_type = data['constructionType']
        budget_encoded = budget_mapping[budget]
        construction_type_encoded = label_enc_type.transform([construction_type])[0]
        features = np.array([[budget_encoded, construction_type_encoded]])
        prediction = model.predict(features)
        material = label_enc_materials.inverse_transform(prediction)[0]
        return jsonify({'prediction': material})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/recommend_phases', methods=['POST'])
def recommend_phases():
    data = request.get_json()
    budget = data.get('budget')
    construction_type = data.get('constructionType')
    if not budget or not construction_type:
        return jsonify({'error': 'Missing parameters'}), 400

    row = df_excel[
        (df_excel['Construction Type'] == construction_type) &
        (df_excel['Budget'] == budget)
    ]
    if row.empty:
        return jsonify({'error': 'No data found for this selection'}), 404

    phases_data = {col: row.iloc[0][col] for col in phase_cols}
    return jsonify({'phases_data': phases_data})

# --- NEW: Check Availability API ---
@app.route('/check_availability', methods=['POST'])
def check_availability():
    data = request.get_json()
    material = data.get('material')
    if not material:
        return jsonify({'error': 'No material provided'}), 400
    available = check_material_in_database(material)
    return jsonify({'available': available})

if __name__ == '__main__':
    app.run(debug=True)
