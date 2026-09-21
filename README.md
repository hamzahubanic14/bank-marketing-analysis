# bank-marketing-analysis

Bank marketing campaign analysis using Python, data preprocessing, exploratory data analysis and machine learning.

# Bank Marketing — Predicting Term Deposit Subscription

CS417 Introduction to Data Mining — project notebook.

## Students

- Abdalkarim Alaraj 
- Muhamed Maglic 
- Hamza Hubanic 

## Project overview

The goal of this project is to predict whether a bank client will subscribe to a term deposit (`y` = yes / no), based on data from a Portuguese bank's direct phone marketing campaigns.

We use the **Bank Marketing** dataset from the UCI Machine Learning Repository (ID 222), which contains around 45,000 clients and 16 attributes (client information, last-contact details, and previous-campaign history).

## Project steps

1. Load and understand the dataset
2. Clean and preprocess the data
3. Exploratory Data Analysis (EDA) with plots
4. Train five classification models
5. K-Means clustering (without using the target variable)
6. Feature importance analysis
7. Compare the models and draw conclusions

### Models used

- Decision Tree
- Naive Bayes
- k-Nearest Neighbors (k-NN)
- Random Forest
- Logistic Regression

Additionally, "balanced" versions of Decision Tree, Random Forest, and Logistic Regression were trained using `class_weight="balanced"` to address class imbalance.

## Data preprocessing

- The target `y` was mapped to a numeric value (yes → 1, no → 0)
- The `duration` column was dropped, since it is only known after the call ends and therefore can't be used for realistic prediction
- In `pdays`, the value `-1` (client never contacted before) was replaced with `0`, and a new binary column `was_contacted_before` was added to preserve that information
- Missing values (`NaN`) in categorical columns were not dropped — one-hot encoding (`pd.get_dummies`) automatically encodes them as zeros across all dummy columns for that feature
- Duplicate rows were removed
- Outliers were not removed, since most represent real clients (very high balance, many contacts) rather than data errors
- Categorical columns were one-hot encoded
- Data was split into training (75%) and test (25%) sets, using `stratify=y`
- Numeric features were scaled (`StandardScaler`) for models that depend on feature scale (k-NN, Logistic Regression)

## Exploratory Data Analysis (EDA)

- The target is imbalanced — most clients did not subscribe
- Subscription rate varies by job (students and retirees are above average)
- Clients whose previous campaign outcome was a success are far more likely to subscribe again
- Subscribers tend to have a slightly higher balance; clients contacted many times in the current campaign are less likely to subscribe
- Correlations between numeric features are mostly weak, except between `pdays`, `previous`, and `was_contacted_before`

## Model results

All models were evaluated using accuracy, precision, recall, F1, and ROC-AUC, along with confusion matrices.

- All models reach a high accuracy of about 89% (except Naive Bayes at 85.6%), but this is misleading due to class imbalance
- **Naive Bayes** has the best F1 (0.41) and recall (0.42) among the original models — best at actually finding subscribers
- **Random Forest** has the best ROC-AUC (0.77) — best at ranking clients for a calling list
- **Logistic Regression** has the highest precision (0.67) but the lowest recall (0.18)
- Decision Tree and k-NN fall in between — high accuracy but low recall (around 0.21–0.23)

### Effect of `class_weight="balanced"`

- Logistic Regression (balanced): recall rises from 0.18 to 0.60, precision drops from 0.67 to 0.26, accuracy drops to about 0.76
- Decision Tree (balanced): recall rises from 0.23 to 0.40, F1 from 0.33 to 0.40, ROC-AUC from 0.66 to 0.71
- Random Forest (balanced): barely changes, F1 even drops slightly (0.33 → 0.30)

Class weighting clearly helps recall but is not a magic fix — it mainly trades precision for recall, and whether the trade is worth it depends on the bank's goal.

## K-Means clustering

- The optimal number of clusters was chosen using the elbow method — `k = 4` was selected
- Clusters differ mainly in age, balance, and contact history
- The cluster of clients contacted in a previous campaign has the highest subscription rate (~20%)
- The cluster with many contacts in the current campaign but little prior history has the lowest rate (~4%)

## Feature importance (Random Forest)

The most important features are `balance`, `age`, and `day_of_week`, followed by `campaign` and `poutcome_success`. A client's financial situation and age are the strongest predictors, while the success of the previous campaign also stands out as a significant signal.

## Conclusion

On an imbalanced dataset, high accuracy can be misleading — most models reached about 89% accuracy but missed the majority of real subscribers. Looking at recall and F1, Naive Bayes was the best of the original models at finding subscribers, while Random Forest had the best ROC-AUC and is best suited for ranking clients. Adding class weights significantly improved recall (Logistic Regression went from finding 18% to 60% of subscribers), at the cost of precision. So the best model depends on the bank's actual goal (maximizing subscribers found vs. minimizing wasted calls).

## Future work

- Compare class weighting with SMOTE oversampling to see which handles the imbalance better
- Tune the decision threshold (instead of the default 0.5) so the bank can choose its own precision/recall trade-off
- Hyperparameter tuning (`GridSearchCV`) for k-NN, Decision Tree, and Random Forest
- Try additional models such as SVM or Gradient Boosting

## Project structure

```
CS417_BankMarketingProject_1.ipynb   # main notebook with the full analysis
bank_marketing.csv                    # local copy of the dataset (generated on run)
```

## Running the project

The notebook automatically downloads the dataset from the UCI repository using the `ucimlrepo` package. Required libraries:

```
numpy, pandas, matplotlib, seaborn, scikit-learn, ucimlrepo
```

After installing the dependencies, simply run all notebook cells in order (Run All).
