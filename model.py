import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv("Housing.csv") #reading the dataset
print(data.info()) #printing the info of the data set
#feature engineering used to increase accuracy
# data['area_bedrooms_interaction'] = data['area'] * data['bedrooms']
# data['area_bathrooms_interaction'] = data['area'] * data['bathrooms']
# data['area_stories_interaction'] = data['area'] * data['stories']
# data['area_parking_interaction'] = data['area'] * data['parking']
# data['bedrooms_bathrooms_interaction'] = data['bedrooms'] * data['bathrooms']
# data['bedrooms_stories_interaction'] = data['bedrooms'] * data['stories']
# data['bedrooms_parking_interaction'] = data['bedrooms'] * data['parking']
# data['stories_parking_interaction'] = data['stories'] * data['parking']
x = data.drop(columns=["price"])  #target column that we will predict
y = data["price"]
#correlation map
datatransformed = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]
numerical_cols = x.select_dtypes(include=["float64", "int64"]).columns


continuous_features = data[["bedrooms", "price"]]  
#features with yes/no data options
categorical_features = data[["basement", "hotwaterheating", "airconditioning"]]  

#creating a histogram with continous features
continuous_features.hist(bins=30, figsize=(10, 5))
plt.suptitle("Distributions of Continuous Features")
#plt.show()

plt.figure(figsize=(12, 10))
plotnumber = 1
for column in continuous_features.columns:
    plt.subplot(2, 2, plotnumber)
    sns.boxplot(y=continuous_features[column])
    plt.title(f"Boxplot of {column}")
    plotnumber += 1  

plt.tight_layout()
#plt.show()

#histogram
plt.figure(figsize=(12, 10))
for i, column in zip(range(len(continuous_features.columns)), continuous_features.columns):
    plt.subplot(2, 2, i + 1)
    plt.hist(continuous_features[column], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(continuous_features[column].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(continuous_features[column].median(), color='green', linestyle='dashed', linewidth=1, label='Median')
    plt.title(f"Overall Histogram of {column}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

plt.suptitle("Overall Distribution of Continuous Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.show()

correlation_matrix = continuous_features.corr()
plt.figure(figsize=(6, 3))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix of Continuous Features")
#plt.show()

plt.figure(figsize=(12, 6))
for i, column in enumerate(continuous_features.columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=data['basement'], y=continuous_features[column])
    plt.title(f"{column} Distribution by basement")
    
plt.tight_layout()
#plt.show()

plt.figure(figsize=(12, 6))
for i, column in enumerate(continuous_features.columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=data['hotwaterheating'], y=continuous_features[column])
    plt.title(f"{column} Distribution by hotwaterheating")
    
plt.tight_layout()
#plt.show()

plt.figure(figsize=(12, 6))
for i, column in enumerate(continuous_features.columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=data['airconditioning'], y=continuous_features[column])
    plt.title(f"{column} Distribution by airconditioning")

plt.tight_layout()
#plt.show()

label = 'price'  

if label in data.columns:
    significant_features = correlation_matrix[label].abs().nlargest(3).index.tolist()  

    plt.figure(figsize=(10, 5))
    for i, feature in enumerate(significant_features[1:], 1):  
        plt.subplot(1, 2, i)
        sns.regplot(x=data[label], y=data[feature], scatter_kws={'alpha': 0.5})
        plt.title(f"{label} vs. {feature} with Regression Line")
        plt.xlabel(label)
        plt.ylabel(feature)

    plt.tight_layout()
    #plt.show()
else:
    print("Target variable not found in the dataset.")

    continuous_features = data[["area", "price", "parking"]]  
categorical_features = data[[ "furnishingstatus", "prefarea"]]  


continuous_features.hist(bins=30, figsize=(10, 5))
plt.suptitle("Distributions of Continuous Features")
#plt.show()

plt.figure(figsize=(12, 10))
plotnumber = 1
for column in continuous_features.columns:
    plt.subplot(2, 2, plotnumber)
    sns.boxplot(y=continuous_features[column])
    plt.title(f"Boxplot of {column}")
    plotnumber += 1  

plt.tight_layout()
#plt.show()

plt.figure(figsize=(12, 10))
for i, column in zip(range(len(continuous_features.columns)), continuous_features.columns):
    plt.subplot(2, 2, i + 1)
    plt.hist(continuous_features[column], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(continuous_features[column].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(continuous_features[column].median(), color='green', linestyle='dashed', linewidth=1, label='Median')
    plt.title(f"Overall Histogram of {column}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

plt.suptitle("Overall Distribution of Continuous Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.show()

correlation_matrix = continuous_features.corr()
plt.figure(figsize=(6, 3))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix of Continuous Features")
#plt.show()

plt.figure(figsize=(12, 6))
for i, column in enumerate(continuous_features.columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=data['parking'], y=continuous_features[column])
    plt.title(f"{column} Distribution by Parking")
    
plt.tight_layout()
#plt.show()

plt.figure(figsize=(12, 6))
for i, column in enumerate(continuous_features.columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=data['furnishingstatus'], y=continuous_features[column])
    plt.title(f"{column} Distribution by Furnishing Status")
    
plt.tight_layout()
#plt.show()

plt.figure(figsize=(12, 6))
for i, column in enumerate(continuous_features.columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=data['prefarea'], y=continuous_features[column])
    plt.title(f"{column} Distribution by Preferred Area")

plt.tight_layout()
#plt.show()

label = 'price'  

if label in data.columns:
    significant_features = correlation_matrix[label].abs().nlargest(3).index.tolist()  

    plt.figure(figsize=(10, 5))
    for i, feature in enumerate(significant_features[1:], 1):  
        plt.subplot(1, 2, i)
        sns.regplot(x=data[label], y=data[feature], scatter_kws={'alpha': 0.5})
        plt.title(f"{label} vs. {feature} with Regression Line")
        plt.xlabel(label)
        plt.ylabel(feature)

    plt.tight_layout()
    #plt.show()
else:
    print("Target variable not found in the dataset.")


continuous_features = data[["bathrooms", "stories", "price"]]  
categorical_features = data[["mainroad", "guestroom"]]  


continuous_features.hist(bins=30, figsize=(10, 5))
plt.suptitle("Distributions of Continuous Features")
#plt.show()

plt.figure(figsize=(12, 10))
plotnumber = 1
for column in continuous_features.columns:
    plt.subplot(2, 2, plotnumber)
    sns.boxplot(y=continuous_features[column])
    plt.title(f"Boxplot of {column}")
    plotnumber += 1  

plt.tight_layout()
#plt.show()

plt.figure(figsize=(12, 10))
for i, column in zip(range(len(continuous_features.columns)), continuous_features.columns):
    plt.subplot(2, 2, i + 1)
    plt.hist(continuous_features[column], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(continuous_features[column].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(continuous_features[column].median(), color='green', linestyle='dashed', linewidth=1, label='Median')
    plt.title(f"Overall Histogram of {column}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

plt.suptitle("Overall Distribution of Continuous Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.show()

correlation_matrix = continuous_features.corr()
plt.figure(figsize=(6, 3))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix of Continuous Features")
#plt.show()

plt.figure(figsize=(12, 6))
for i, column in enumerate(continuous_features.columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=data['mainroad'], y=continuous_features[column])
    plt.title(f"{column} Distribution by mainroad")
    
plt.tight_layout()
#plt.show()

plt.figure(figsize=(12, 6))
for i, column in enumerate(continuous_features.columns, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=data['guestroom'], y=continuous_features[column])
    plt.title(f"{column} Distribution by guestroom")

plt.tight_layout()
#plt.show()

label = 'price'  

if label in data.columns:
    significant_features = correlation_matrix[label].abs().nlargest(3).index.tolist()  

    plt.figure(figsize=(10, 5))
    for i, feature in enumerate(significant_features[1:], 1):  
        plt.subplot(1, 2, i)
        sns.regplot(x=data[label], y=data[feature], scatter_kws={'alpha': 0.5})
        plt.title(f"{label} vs. {feature} with Regression Line")
        plt.xlabel(label)
        plt.ylabel(feature)

    plt.tight_layout()
    #plt.show()
else:
    print("Target variable not found in the dataset.")


#unproccessed data being converted to cleaned data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")), # missing values being dealt with
    ("scaler", StandardScaler()) #standard scaler to scale large values
])

#to clean string data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), #adds most repeated values in place of missing values
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")) #assigns number to yes and no as 1 and 0
])

# takes the pipelines and makes final changes
transformer = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, datatransformed)
    ],
    remainder="passthrough"
)

#test train data split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=42)

#selected supervised ML model being used
pipeline = Pipeline([
    ("preprocessor", transformer),
    ("regressor", LinearRegression())
])

#fitting the test data to the linear regression algorithm to train it
pipeline.fit(xtrain, ytrain)
#using the model to predict test values
predictions = pipeline.predict(xtest)
#calculating mean error
mae = mean_absolute_error(ytest, predictions)
#calculating the accuracy score
r2_score = pipeline.score(xtest, ytest)


print("Model: Linear Regression")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2_score:.4f}")

#adding the predicted values pipeline to the pkl file
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
