from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the saved model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

def generate_plot(data, plot_type):
    buffer = BytesIO()

    if plot_type == 'bedrooms':
        plt.figure()
        plt.hist(data['bedrooms'], bins=30, color='skyblue', edgecolor='black')
        plt.title("Distribution of Bedrooms")
        plt.tight_layout()
    elif plot_type == 'price_boxplot':
        plt.figure()
        plt.boxplot(data['price'])
        plt.title("Boxplot of Price")
        plt.tight_layout()
    elif plot_type == 'price_vs_area':
        plt.figure()
        plt.scatter(data['area'], data['price'], alpha=0.5)
        plt.title("Price vs. Area")
        plt.tight_layout()
    elif plot_type == 'correlation':
        plt.figure(figsize=(8, 6))
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()

    plt.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()
    return encoded_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = request.form.to_dict()
    try:
        inputs_transformed = []
        for col, value in inputs.items():
            if col in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
                inputs_transformed.append(1 if value.lower() == 'yes' else 0)
            elif col == 'furnishingstatus':
                inputs_transformed.append({'furnished': 1, 'semi-furnished': 2, 'unfurnished': 3}[value.lower()])
            else:
                inputs_transformed.append(float(value))

        input_df = pd.DataFrame([inputs_transformed], columns=[
            "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning",
            "prefarea", "furnishingstatus", "area", "bedrooms", "bathrooms", "stories", "parking"
        ])
        prediction = model.predict(input_df)[0]
        return jsonify({"price": f"Rs {prediction:,.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/graphs')
def graphs():
    data = pd.read_csv("Housing.csv")
    categorical_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]
    for col in categorical_columns:
        data[col] = data[col].map({'yes': 1, 'no': 0})

    image_data_bedrooms = generate_plot(data, 'bedrooms')
    image_data_price_boxplot = generate_plot(data, 'price_boxplot')
    image_data_price_vs_area = generate_plot(data, 'price_vs_area')
    image_data_correlation = generate_plot(data, 'correlation')

    return render_template('graphs.html', 
                           image_data_bedrooms=image_data_bedrooms,
                           image_data_price_boxplot=image_data_price_boxplot,
                           image_data_price_vs_area=image_data_price_vs_area,
                           image_data_correlation=image_data_correlation)

if __name__ == '__main__':
    app.run(debug=True)
