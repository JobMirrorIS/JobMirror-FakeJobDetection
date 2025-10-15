
# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack


# 2. Load the Cleaned Dataset
df = pd.read_csv('/content/cleaned_final_dataset.csv')  # adjust path if needed

# 3. Separate Features (X) and Target (y)

X = df.drop('fraudulent', axis=1)  # 'fraudulent' is our target column
y = df['fraudulent']


# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Fill Missing Text Values
# Replace NaNs in text columns with empty strings
text_cols = ['description', 'requirements']  # text-based columns
for col in text_cols:
    X_train[col] = X_train[col].fillna("")
    X_test[col] = X_test[col].fillna("")

# 6. TF-IDF for Text Columns
tfidf_desc = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_req = TfidfVectorizer(max_features=3000, stop_words='english')

X_train_desc = tfidf_desc.fit_transform(X_train['description'])
X_test_desc = tfidf_desc.transform(X_test['description'])

X_train_req = tfidf_req.fit_transform(X_train['requirements'])
X_test_req = tfidf_req.transform(X_test['requirements'])

# 7. Process Numeric & Categorical Columns
# Select other (non-text) features
other_features = [
    'telecommuting',
    'has_company_logo',
    'has_questions',
    'employment_type',
    'required_experience',
    'required_education',
    'industry',
    'function',
    'country',
    'state',
    'has_salary'
]

X_train_other = X_train[other_features]
X_test_other = X_test[other_features]

# Define ColumnTransformer for other features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         ['employment_type', 'required_experience', 'required_education', 'industry', 'function', 'country', 'state']),
        ('num', StandardScaler(),
         ['telecommuting', 'has_company_logo', 'has_questions', 'has_salary'])
    ]
)

X_train_other_transformed = preprocessor.fit_transform(X_train_other)
X_test_other_transformed = preprocessor.transform(X_test_other)
# 8. Combine TF-IDF + Other Features
X_train_transformed = hstack([X_train_desc, X_train_req, X_train_other_transformed])
X_test_transformed = hstack([X_test_desc, X_test_req, X_test_other_transformed])

print("Feature Engineering Completed!")
print("Training feature matrix shape:", X_train_transformed.shape)
print("Testing feature matrix shape:", X_test_transformed.shape)


# Logistic Regression Model – Job Mirror

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Initialize the model
logreg = LogisticRegression(class_weight='balanced', max_iter=500)

# 2. Train the model
logreg.fit(X_train_transformed, y_train)

# 3. Predict
y_pred = logreg.predict(X_test_transformed)
y_proba = logreg.predict_proba(X_test_transformed)[:, 1]

# 4. Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\n Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

print("\n ROC-AUC Score:", round(roc_auc_score(y_test, y_proba), 4))

# 5. Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,5))
plt.plot(fpr, tpr, label="Logistic Regression (AUC = {:.4f})".format(roc_auc_score(y_test, y_proba)))
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Initialize Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# 2. Train
rf_model.fit(X_train_transformed, y_train)

# 3. Predict
y_pred_rf = rf_model.predict(X_test_transformed)
y_proba_rf = rf_model.predict_proba(X_test_transformed)[:, 1]

# 4. Evaluate
print("Classification Report (Random Forest):\n")
print(classification_report(y_test, y_pred_rf))

print("Confusion Matrix:")
conf_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_rf, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n ROC-AUC Score:", round(roc_auc_score(y_test, y_proba_rf), 4))

# 5. ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
plt.figure(figsize=(8,5))
plt.plot(fpr_rf, tpr_rf, label="Random Forest (AUC = {:.4f})".format(roc_auc_score(y_test, y_proba_rf)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.grid()
plt.show()


