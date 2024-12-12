# Remote Experiment tracking
import mlflow
import mlflow.sklearn 
from sklearn.datasets import load_wine 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 
import dagshub

# Set tracking url 
mlflow.set_tracking_uri("http://127.0.0.1:5000")
dagshub.init(repo_owner='HiteshRam666', repo_name='MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/HiteshRam666/MLFlow.mlflow")

# Load wine dataset 
wine = load_wine()
X = wine.data
y = wine.target

# Train Test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Params for RF Model 
max_depth = 10
n_estimators = 5

# Set experiment name
mlflow.set_experiment("Experiment1_MLFlow")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Creating a ConfusionMatrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix")
    #Save plot 
    plt.savefig("Confusion-Matrix.png")

    # Log artifacts using MLFlow
    mlflow.log_artifact("Confusion-Matrix.png")
    mlflow.log_artifact(__file__)

    # Add tags
    mlflow.set_tags({'Author': 'hitesh', 'project':'Wine classification'})

    # Log the model
    mlflow.sklearn.log_model(rf, "RandomForest-Model")


    print(f"Accuracy is : {accuracy}")
    

