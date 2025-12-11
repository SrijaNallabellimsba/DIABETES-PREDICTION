# **Diabetes Prediction using Machine Learning**

End-to-end ML project predicting diabetes using Glucose, BMI, Age, Pregnancies and engineered features.

Focus: high Recall, low false negatives, strong AUC performance.

## Project Workflow:

Load data \& explore
Clean missing/zero medical values
Outlier handling
Feature engineering
Train–test split
SMOTE balancing
Model training (7 models)
Hyperparameter tuning (GridSearchCV)
Evaluation: Accuracy, Recall, Precision, F1, AUC
Confusion matrices, ROC curves

## Final Model Results:

Model Performance Summary:
    

  Model	                Accuracy   Recall	  Precision	    F1	      AUC
6	LightGBM	          0.840	    0.90	  0.803571	0.849057	0.89680
4	Random Forest	      0.820	    0.91	  0.771186	0.834862	0.87840
0	KNN	                  0.815	    0.93	  0.756098	0.834081	0.85645
1	SVM	                  0.800	    0.89	  0.754237	0.816514	0.85000
5	XGBoost	              0.795	    0.92	  0.736000	0.817778	0.85460
3	Decision Tree	      0.750	    0.79	  0.731481	0.759615	0.75000
2	Logistic Regression	  0.745	    0.75	  0.742574	0.746269	0.82300

### Key Insights

Glucose, BMI and Age are strongest predictors
LightGBM best balance of recall + AUC
KNN highest recall → lowest false negatives
Outlier treatment + SMOTE greatly improved model stability

### Technologies Used

Python
Scikit-Learn
LightGBM
XGBoost
Seaborn / Matplotlib

### How to Run

pip install -r requirements.txt

jupyter notebook

Open Diabetes\_Prediction.ipynb and run all cells.
