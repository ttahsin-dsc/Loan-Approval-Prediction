# Loan Approval Prediction Using Machine Learning

### Logistic Regression | Data Preprocessing | Explainable AI | Responsible Model Evaluation

A supervised machine learning project that predicts **historical loan application approval outcomes** using applicants' financial information. The analysis covers raw-data validation, missing-value treatment, feature selection, stratified cross-validation, model training and evaluation, with attention to **interpretability, data leakage and fairness**.

The goal was not simply to maximise predictive accuracy, but to build a transparent baseline and examine the practical consequences of incorrect predictions in a financial decision-making context.

> **Project context:** Developed as part of my MSc Data Science coursework. This is an educational prototype trained to reproduce patterns in previous decisions—not a credit-risk or default-prediction system, and **not suitable for automated lending decisions**.

## Project at a glance

| Item | Details |
|---|---|
| Problem | Binary classification: approved (`1`) versus not approved (`0`) |
| Dataset | Historical loan application records (`previousApplicants.csv`) |
| Original data | 13,731 applications, 7 columns |
| Analysis-ready data | 13,500 applications after cleaning |
| Selected predictors | `Income`, `Outgoings` |
| Algorithm | Regularised Logistic Regression (scikit-learn) |
| Validation | Stratified 80/20 train–test split; 10-fold cross-validation for feature comparison |
| Test accuracy | **63.89%** |
| ROC-AUC | **0.704** |

## Business problem

Loan application decisions require careful assessment of financial information, consistency and accountability. In this project, I investigated whether a relatively simple, interpretable model could learn patterns from **past approval outcomes** and predict an outcome for previously unseen applications.

I focused on three questions:

1. How can an imperfect dataset be cleaned and prepared without leaking information from the test set into training?
2. Can a small number of financially relevant predictors produce a useful and explainable classification baseline?
3. What do the model's errors reveal about the risks of using predictions to support lending decisions?

## Tools and techniques

**Python**, **pandas**, **NumPy**, **Matplotlib** and **scikit-learn**.

Methods used: data profiling, validation of numerical values, duplicate detection, missing-value analysis, IQR-based outlier checks, median imputation, standardisation, stratified sampling, 10-fold cross-validation, Logistic Regression, confusion matrices, precision, recall, F1-score, ROC curves and ROC-AUC.

## 1. Data understanding and preparation

The source dataset contained seven fields: `ID`, `Age`, `Income`, `Postcode`, `Gender`, `Outgoings` and `Approved`.

The preprocessing workflow included:

- **Schema and target validation:** inspected data types and missing values; converted the historical `Approved` labels from `TRUE`/`FALSE` to `1`/`0`.
- **Consistency checks:** standardised text formatting in `Gender` and `Postcode`; converted numeric columns to numeric types.
- **Invalid values:** identified **628 implausible `Outgoings` entries** at approximately `1e24` and marked them as missing; also checked age ranges and negative income.
- **Duplicate and unusable record removal:** removed exact duplicates and rows without enough usable financial information, reducing the dataset from **13,731 to 13,500 records**. Repeated applicant IDs were not removed solely for sharing an ID.
- **Exploratory assessment:** inspected distributions and checked potential outliers using the interquartile range (IQR) method.
- **Leakage-aware preprocessing:** fitted median imputation and `StandardScaler` using **training data only**, then applied those fitted transformations to the held-out test data.

After row-level cleaning, **603 `Income`** and **397 `Outgoings`** values remained missing and were handled using training-set median imputation.

### Class balance and feature distributions

![Distribution of historical loan approval outcomes](images/Figure_1_loan_Approval_outcomes.png)

After cleaning, the dataset contained **7,882 approved** and **5,618 not-approved** applications—an approval share of approximately **58.4%**. This imbalance is one reason accuracy alone is not sufficient to evaluate the model.

![Histograms of age, income and outgoings](images/Figure_2_distributionof_numericalfeatures.png)

![Boxplots of age, income and outgoings](images/Figure_3_boxplots_of_numericalfeatures.png)

The histograms and boxplots helped assess ranges, distributions and potentially invalid observations before modelling.

## 2. Model selection and feature choices

I selected **Logistic Regression** because it provides class probabilities and coefficients that can be interpreted more readily than many complex models. I considered a Decision Tree as an alternative: it can capture non-linear relationships but can also overfit or become unstable without appropriate controls.

I compared two candidate feature sets using **stratified 10-fold cross-validation on the training set**. The evaluation pipeline included median imputation, scaling and Logistic Regression within each fold.

| Candidate features | Mean CV accuracy |
|---|---:|
| Income + Outgoings | **64.31%** |
| Age + Income + Outgoings | **64.86%** |

Adding `Age` improved mean cross-validation accuracy by only about **0.55 percentage points**. I therefore selected **`Income` and `Outgoings`** for the final model, prioritising a simpler financial feature set while avoiding unnecessary use of a protected characteristic.

**Features excluded from the final model:**

- `ID` — an identifier rather than a meaningful predictive financial variable.
- `Gender` — a sensitive attribute with potential discrimination concerns.
- `Postcode` — a possible proxy for demographic and socioeconomic characteristics.
- `Age` — offered only a small increase in cross-validation accuracy and raises additional fairness considerations.

> Excluding protected or sensitive attributes **does not establish fairness**. Financial variables and past lending decisions can still reflect historical bias. No formal subgroup fairness audit was performed in this project.

## 3. Model training and results

The cleaned data was split using a **stratified 80/20 train–test split**, with `random_state=50`. The final Logistic Regression model was trained using the two standardised financial features. Predictions used the default **0.50 probability threshold**.

### Performance on the held-out test set (2,700 applications)

| Metric | Result |
|---|---:|
| Training accuracy | 64.39% |
| **Test accuracy** | **63.89%** |
| Precision — approved class | 66.56% |
| Recall — approved class | 76.65% |
| F1-score — approved class | 71.25% |
| **ROC-AUC** | **0.704** |

Training and testing accuracies were close, suggesting little evidence of overfitting in this evaluation. However, the overall classification performance remains **moderate**.

### Confusion matrix

![Logistic Regression confusion matrix](images/confusion_matrix.png)

| Actual outcome | Predicted not approved | Predicted approved |
|---|---:|---:|
| Not approved | **517** | **607** |
| Approved | **368** | **1,208** |

The classifier correctly identified 1,208 historically approved applications, but also predicted approval for **607 historically not-approved applications**. It missed 368 historically approved applications. These disagreements with historical decisions matter: a false approval prediction may warrant additional underwriting checks, while a false rejection prediction may disadvantage an applicant. **Neither error type proves actual creditworthiness or default risk**, because the target represents past decisions, not repayment outcomes.

### ROC curve

![Logistic Regression ROC curve](<images/Logistic regression ROC curve.png>)

An **ROC-AUC of 0.704** indicates that the model separates the two historical outcome classes better than random ranking, but not reliably enough for consequential decisions without further validation.

### Coefficients and interpretability

![Logistic Regression coefficients](<images/Logistic regression coefficients.png>)

| Standardised predictor | Fitted coefficient | Odds ratio |
|---|---:|---:|
| Income | +0.741 | 2.097 |
| Outgoings | −0.049 | 0.952 |

Within the fitted model, higher income was associated with higher predicted odds of historical approval, whereas higher outgoings had a much smaller negative association. Because features were standardised, these coefficients correspond to a **one-standard-deviation change** in a predictor, holding the other constant. They describe model associations, **not causal effects**.

## 4. Interpretation and limitations

The project demonstrated that a transparent baseline can be developed with a reproducible preprocessing and evaluation workflow. It also highlighted several limitations:

- **Moderate predictive performance:** the model makes too many classification errors for use in real lending decisions.
- **False positives:** 607 applications that were historically not approved were predicted as approved. The operating threshold should reflect carefully defined business and consumer risks, not accuracy alone.
- **Limited predictors:** income and outgoings do not capture loan amount, existing debt, repayment history, employment circumstances or other relevant affordability information.
- **Historical labels:** predicting previous approval decisions is different from predicting whether a borrower will repay a loan.
- **Fairness not established:** excluding demographic features reduces direct use of sensitive information but does not remove proxy discrimination or historical bias.
- **Repeated IDs:** the code permits multiple records with the same applicant identifier; a future evaluation should check for applicant overlap between training and test sets to avoid overstating generalisation to new people.

## 5. Potential improvements

For a future iteration, I would:

1. **Tune the decision threshold** against an explicit cost framework, rather than automatically applying 0.50.
2. **Compare additional supervised models**, such as regularised Decision Trees or ensemble methods, under the same evaluation strategy.
3. **Engineer financial features**, such as an income-to-outgoings ratio or disposable-income estimate, and assess their added value using cross-validation.
4. **Add more relevant and appropriately sourced data**, especially loan amount and repayment-related variables, with suitable data governance.
5. **Evaluate group-level model performance and fairness**, alongside explainability, calibration and robustness checks.
6. **Use applicant-group-aware splitting** if repeated IDs represent the same applicants, and test generalisation on later or genuinely unseen records where available.


## How to run

**Requirements:** Python 3 and the libraries `numpy`, `pandas`, `matplotlib` and `scikit-learn`.

1. Clone or download this repository.
2. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

3. From the repository root, run:

   ```bash
   python loanapplication.py
   ```

The script expects `previousApplicants.csv` in the **current working directory** and prints data-quality checks, feature-set validation scores, model metrics, coefficients and a sample of predictions. It also displays the charts.

## Responsible use

This project is an **academic demonstration** of supervised classification and model assessment. Approval predictions are not equivalent to credit-risk estimates. Its outputs should not be used to automatically accept or reject loan applications. A production financial decision-support system would require far stronger validation, appropriate oversight, fairness assessment, privacy safeguards and regulatory review.

## Skills demonstrated

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Data Cleaning` · `Exploratory Data Analysis` · `Feature Selection` · `Cross-Validation` · `Logistic Regression` · `Model Evaluation` · `Explainable Machine Learning` · `Responsible AI`

