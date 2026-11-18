import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#^^PART 1 - DATA PREPROCESSING STAGE^^

#load dataset
file_name = "previousApplicants.csv"

dataset = pd.read_csv(
    file_name,
    dtype={"ID": "string"}, #ID  is used here as an identifier instead of number hence loaded as text
    na_values=["null","NULL","Null",""]
)

dataset.columns = dataset.columns.str.strip() #remove space from column names

print("\nDataset loaded successfully")

#check the data structure
print("\nFirst 10 rows:")
print(dataset.head(10))

print("\nDataset shape:",dataset.shape)
print("\nColumn names:",dataset.columns)
print("\nData types:",dataset.dtypes)
print("\nInitial Missing values:",dataset.isnull().sum())
print("\nNumber of duplicated rows:",dataset.duplicated().sum())


numerical_columns = ["Age", "Income", "Outgoings"]

#Data type conversion
for column in numerical_columns:
    dataset[column] = pd.to_numeric(dataset[column],errors="coerce")

#standardising text format of both categorical text columns 
dataset["Gender"] = (dataset["Gender"].astype("string").str.strip().str.lower())
dataset["Postcode"] = (dataset["Postcode"].astype("string").str.strip().str.upper())

#converting boolean values of 'approved' column from true/false to  binary 1/0
approval_values = {
    "TRUE" : 1,
    "FALSE": 0,
}

dataset["Approved"] = (
    dataset["Approved"].astype("string").str.strip().str.upper().map(approval_values)
)

print("\nApproved values after mapping:") #1 is approved,0 is rejected
print(dataset["Approved"].value_counts(dropna=False))

print("\nGender categories found in the dataset:")
print(dataset["Gender"].value_counts(dropna=False))


#Data validation stage - handling invalid values in 'outgoing' column

print("\nLargest Outgoing values:")
print(
    dataset["Outgoings"]
    .sort_values(ascending=False)
    .head(10))

invalid_outgoings = ((dataset["Outgoings"] < 0) | (dataset["Outgoings"] >=1e24))#in 'outgoings' column extreme values like 1E+24 were observed

print("\nNumber of invalid outgoings values such as 1E+24:",invalid_outgoings.sum())
if invalid_outgoings.any():
    print("\nInvalid outgoings records:")
    print(dataset.loc[invalid_outgoings,["ID","Outgoings"]])
    dataset.loc[invalid_outgoings, "Outgoings"] = np.nan #replace invalid outgoing values with Nan

    print("\nInvalid outgoings values were replaced with Nan")
else:
    print("\nNo invalid outgoings values were found")


#handling unusually high or low values in 'ages' column like people under 18 and above 100 can't apply for loans 
print("\nHighest Age Values:")
print(dataset["Age"]
      .sort_values(ascending=False)
      .head(10)
)

invalid_age = ((dataset["Age"] < 18) | (dataset["Age"] > 100))

print("\nInvalid Age records:",invalid_age.sum())
if invalid_age.any():
    print("\nInvalid Age records:")
    print(dataset.loc[invalid_age, ["ID","Age"]])
    dataset.loc[invalid_age,"Age"] = np.nan
else:
    print("\nNo invalid age values were found")

#check and handle negative income values
negative_income = dataset["Income"] < 0
print("\nNumber of negative values of Income:",negative_income.sum())

if negative_income.any():
    print("\nNegative income records:")
    print(dataset.loc[negative_income, ["ID","Income"]])
    dataset.loc[negative_income,"Income"] = np.nan
else:
    print("\nNo negative Income values were found")


#Data cleaning part - remove records that can't be used for model training later
rows_before_cleaning = len(dataset)

dataset.dropna(subset=["Approved"],inplace=True) #only checking approved columns for missing values
dataset["Approved"] = dataset["Approved"].astype(int) #converting 'Approved' to integer

duplicate_rows = dataset.duplicated().sum() 
dataset.drop_duplicates(inplace=True) #remove exact duplicate rows but keeping records with same ID where other application details different

dataset.dropna(subset=["Income", "Outgoings"],how="all",inplace=True) #remove rows where both 'income' and 'outgoings' are missing together

dataset.reset_index(drop=True,inplace=True)

rows_after_cleaning = len(dataset)
print("\nRepeated ID occurance:", dataset["ID"].dropna().duplicated().sum())

print("\nRows before cleaning: ",rows_before_cleaning)
print("\nTotal duplicate rows removed: ",duplicate_rows)
print("\nRows after cleaning: ", rows_after_cleaning)
print("\nTotal rrows removed: ", rows_before_cleaning - rows_after_cleaning)

#check the cleaned dataset
print("\nCleaned dataset shape: ",dataset.shape)
print("\nMissing values after correcting invalid entries: ", dataset.isnull().sum())
print("\nNumerical Summary: ", dataset[numerical_columns].describe())

print("\nApproved counts: ")
print(dataset["Approved"].value_counts().sort_index())
print("\nApproved count percentage:")
print(dataset["Approved"].value_counts(normalize=True).sort_index()*100)

print("\nNumber of different types of postcode:")
print(dataset["Postcode"].nunique())

#confirming both approved options are available
if dataset["Approved"].nunique() != 2:
    raise ValueError(
        "Approved must contain both 0 and 1"
    )

#checking outliers using IQR method
#code adapted from CO7052 lecture slides, Mpofu, 2026
print("\nPossible outliers identified by IQR method: ")
for column in numerical_columns:
    values = dataset[column].dropna()

    Q1=values.quantile(0.25)
    Q3=values.quantile(0.75)
    IQR = Q3 - Q1

    #define the outlier boundaries
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    #count values outside boundaries
    outlier_count = ((values < lower_bound) | (values > upper_bound)).sum()


    print(f"\n{column}:"
          f"lower bound = {lower_bound:.2f},"
          f"upper bound = {upper_bound: .2f},"
          f"possible outliers = {outlier_count}"
    )#end of adapted code

#^^Data Visualisations - after initial data processing^^

#approved outcome distribution
approval_counts = dataset["Approved"].value_counts().sort_index()

approval_counts.index = ["Not Approved", "Approved"]

approval_counts.plot(kind="bar",figsize=(10,8))

plt.title("Distribution of Loan Approval Outcomes")
plt.xlabel("Application Outcome")
plt.ylabel("Number of Applications")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#Histograms of numerical columns = age, income, outgoings
dataset[numerical_columns].hist(bins=20,figsize=(14,6))

plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.show()

#Boxplot of numerical variables

dataset[numerical_columns].plot(kind="box",subplots=True,layout=(1,3),figsize=(14,6))

plt.suptitle("Boxplots of Numerical features")
plt.tight_layout()
plt.show()



#^^Part 2 - Training and testing^^ 
#create training and testing sets

candidate_features = ["Age", "Income", "Outgoings"] #predictor variables

X_all = dataset[candidate_features]
y = dataset["Approved"]

X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all,
    y,
    test_size=0.20,
    random_state=50,
    stratify=y
)

print("\nTraining set dimensions: ",X_train_all.shape)
print("\nTesting set dimensions: ",X_test_all.shape)
print("\nTraining target distribution:",y_train.value_counts().sort_index())
print("\nTesting target distributions: ", y_test.value_counts().sort_index())
print("\nTraining target percentages",y_train.value_counts(normalize=True).sort_index()*100)
print("\nTesting target percentages",y_test.value_counts(normalize=True).sort_index()*100)


#compare between two combinations of candidate feature sets
feature_sets = {"Income and Outgoings" : ["Income","Outgoings"],
                "Age, Income and Outgoings" : ["Age","Income","Outgoings"]}

stratified_kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=50)

feature_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

feature_set_results = []
for feature_set_name, feature_columns in feature_sets.items():
    feature_scores = cross_val_score(
        feature_model,
        X_train_all[feature_columns],
        y_train,
        cv=stratified_kfold,
        scoring="accuracy"
    )

    feature_set_results.append({
        "Feature Set": feature_set_name,
        "Mean accuracy": feature_scores.mean(),
        "Accuracy SD": feature_scores.std()
    })

feature_set_results = pd.DataFrame(feature_set_results)

print("\nCandidate feature set comparison: ")
print(feature_set_results.round(4).to_string(index=False))

#Final feature decision selected for training and testing:

features = ["Income", "Outgoings"]

X_train = X_train_all[features]
X_test = X_test_all[features]

print("\nFeatures selected for the final model:", features)
print("\nFeatures excluded from the final model:", ["ID", "Age", "Postcode", "Gender"])


#Median imputation
imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=features,
    index=X_train.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=features,
    index=X_test.index
)

print("\nMedian values used for imputation:", pd.Series(imputer.statistics_,index=features))
print("\nMissing values after imputation:")
print("Training set:")
print(X_train_imputed.isnull().sum())
print("Testing set:")
print(X_test_imputed.isnull().sum())

#Scaling the features for selected ML model - Logistic Regression
scale = StandardScaler()


X_train_scaled = pd.DataFrame(
    scale.fit_transform(X_train_imputed),
    columns=features,
    index=X_train_imputed.index
)
X_test_scaled = pd.DataFrame(
    scale.transform(X_test_imputed),
    columns=features,
    index=X_test_imputed.index
)

print("\nScaled training records:")
print(X_train_scaled.head(10))



#^^Part 3 - Train selected ML model - Logistic Regression

#code adapted from LogisticRegression documentation (Scikit-learn developer, n.d.)
logmodel = LogisticRegression(max_iter=1000) #creating the ML model
logmodel.fit(X_train_scaled,y_train) #to train the model

all_probabilities = logmodel.predict_proba(X_test_scaled) #calculate the probabilities for both classes
approval_probabilities = all_probabilities[:, 1] #for every test, return predicted probabilities of class 1(approved)
#end of adapted code

#converting probabilities into binary predictions with default 0.5 threshold
logistic_predictions = [
    1 if probability >= 0.5 else 0 
    for probability in approval_probabilities
]

print("\nPredicted approval probabilities(first 10 rows only):")
print(approval_probabilities[:10])

print("\nPredicted classes(first 10 rows only):")
print(logistic_predictions[:10])

#Evaluating the model

logistic_training_predictions = logmodel.predict(X_train_scaled)

logistic_training_accuracy = accuracy_score(y_train,logistic_training_predictions)
logistic_testing_accuracy = accuracy_score(y_test,logistic_predictions)

logistic_precision = precision_score(y_test,logistic_predictions,zero_division=0)
logistic_recall = recall_score(y_test,logistic_predictions,zero_division=0)
logistic_f1 = f1_score(y_test,logistic_predictions,zero_division=0)
logistic_matrix = confusion_matrix(y_test, logistic_predictions, labels=[0,1])
logistic_auc = roc_auc_score(y_test,approval_probabilities)

print("\nLogistic Regression results: ")
print("\nTraining accuracy: ", round(logistic_training_accuracy, 4))
print("\nTesting accuracy: ", round(logistic_testing_accuracy, 4))
print("\nPrecision for Approved: ",round(logistic_precision, 4))
print("\nRecall for Approved: ",round(logistic_recall, 4))
print("\nF1 score for Approved",round(logistic_f1, 4))
print("\nROC-AUC: ",round(logistic_auc, 4))
print("\nConfusion matrix: ")
print(logistic_matrix)
print("\nClassification report: ")
print(
    classification_report(
        y_test,
        logistic_predictions,
        labels=[0,1],
        target_names=[
            "Not Approved",
            "Approved"
        ],
        zero_division=0
    )
)
true_negative = logistic_matrix[0,0]
false_positive = logistic_matrix[0,1]
false_negative = logistic_matrix[1,0]
true_positive = logistic_matrix[1,1]

print("\nCorrectly predicted as not approved:")
print(true_negative)

print("\nIncorrectly predicted as approved:")
print(false_positive)

print("\nIncorrectly predicted as not approved: ")
print(false_negative)

print("\nCorrectly predicted as approved: ")
print(true_positive)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    logistic_predictions,
    display_labels=["Not Approved", "Approved"]
)
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()


#logistic regression coefficients and odds ratios
coefficient_results = pd.DataFrame({
    "Feature": features,
    "Coefficient": logmodel.coef_[0],
    "Odds Ratio": np.exp(logmodel.coef_[0])
})

print("\nLogistic Regression coefficients and odds ratios:")
print(coefficient_results)

coefficient_results.set_index("Feature")["Coefficient"].plot(
    kind="bar",
    figsize= (10,7)
)

plt.axhline(0)
plt.title("Logistic Regression Coefficients")
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#code adapted from roc_curve documentation (Scikit-learn developer, n.d.)
logistic_fpr, logistic_tpr, thresholds = roc_curve(y_test, approval_probabilities) #end of adapted code

#to plot the logistic regression ROC curve 
plt.figure(figsize=(10,8))
plt.plot(
    logistic_fpr,
    logistic_tpr,
    label="Logistic Regression AUC = " + str(round(logistic_auc, 3))
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    label="Random Classification"
)

plt.title("Logistic Regression ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show() 

#summarising the final test predictions
prediction_results = dataset.loc[X_test.index,["ID"]].copy()
prediction_results["Actual Approved"] = y_test
prediction_results["Predicted Approved"] = logistic_predictions
prediction_results["Approval Probability"] = approval_probabilities

print("\nFinal prediction results(first 10 rows only): ")
print(prediction_results.head(10))