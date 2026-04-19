import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("data/heart.csv")


# Split
# Use correct target column
X = df.drop(["target_binary", "num"], axis=1)  
y = df["target_binary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODELS TO TUNE & EVALUATE ----------------
# We define multiple models and grids for hyperparameter optimization
model_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    },
    "Support Vector Machine": {
        "model": SVC(probability=True, random_state=42), # probability=True is needed for predict_proba in app.py
        "params": {
            "C": [0.1, 1, 10, 50],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        }
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    }
}

best_model = None
best_score = 0
best_name = ""

print("Evaluating and tuning models. This may take a moment...\n")

for name, mp in model_params.items():
    # Use GridSearchCV to find the best hyperparameters
    clf = GridSearchCV(mp["model"], mp["params"], cv=5, n_jobs=-1, scoring='accuracy')
    clf.fit(X_train, y_train)
    
    # Evaluate best estimator on test data
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    best_cv_score = clf.best_score_
    
    print(f"{name}:")
    print(f"  Best Params: {clf.best_params_}")
    print(f"  Best CV Accuracy (Training validation): {best_cv_score:.3f}")
    print(f"  Test Accuracy (Unseen data): {test_acc:.3f}\n")
    
    if best_cv_score > best_score:
        best_score = best_cv_score
        best_model = clf.best_estimator_
        best_name = name

print(f"✅ Best Overall Model Selected: {best_name} with CV Score: {best_score:.3f}")

# Save the best model and scaler
pickle.dump(best_model, open("model/best_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
print("The best model and scaler have been saved successfully.")