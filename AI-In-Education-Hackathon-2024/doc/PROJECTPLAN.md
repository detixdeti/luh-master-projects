# Bias Analysis towards Fair AI by Utilizing Student Performance Dataset

## Project Overview
We aim to analyze and demonstrate the correlation between student performance and factors such as ethnicity/race and parental educational background, highlighting educational inequalities. This analysis will help showcase potential disparities and propose insights for educators to foster fairness in education.

## Section 1: Problem Definition
- **Problem Statement:**
  - **Challenge:** Identify potential inequalities in student performance based on race/ethnicity, parental education and gender.
  - **Consequences:** Addressing unfair opportunities is crucial for creating a more equitable learning environment.
  - **Target Group:** Educators, researchers, and students interested in educational equity.

- **Problem Significance:**
  - **Importance:** Uncovering disparities can drive improvements in educational practices to ensure fairness.
  - **Challenges:**
    - Selecting a suitable ML model to identify patterns.
    - Effectively cleaning and preparing the dataset.
    - Avoiding misinterpretation of correlation as causation and ensuring robust analysis.

- **Target Questions:**
  1. Are there disparities in performance among different groups?
  2. Could these disparities indicate unfair opportunities?
  3. How can educators use this data to create fairer educational practices?

## Section 2: Goals and Objectives
- **Goals and Outcomes:**
  - **Objective:** Showcase and validate inequalities in student performance.
  - **Impact:** Motivate educators to assess and improve their practices to enhance fairness.

- **Expected Impact:**
  - **Verification:** Use metrics such as accuracy, precision, ROC-AUC, and MSE/MAE to validate findings.
  - **Evaluation Strategy:** Employ K-Fold Cross-Validation and statistical tests to assess the results.

- **Requirements:**
  1. Proper dataset division (train, validate, test).
  2. Data pre-processing if necessary.
  3. Application of classification or regression models.
  4. Monitoring for over-/underfitting.
  5. Providing interpretable plots.
  6. Ensuring reproducibility (seeds, documented software/hardware).

## Section 3: Methodology
- **Data Source:** Kaggle - Student Performance Prediction Dataset ([Link](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics)).

- **Methodology:**
  - **Model Selection:** Experiment with models like Random Forests, Decision Trees, Linear/Logistic Regression, Neural Networks, and K-Means Clustering.
  - **Data Splitting:** Apply K-Fold Cross-Validation (70-20-10 split).
  - **Metrics for Evaluation:** Accuracy, Precision, ROC-AUC, MSE/MAE.
  - **Visualization:** Utilize plots (e.g., histograms, scatterplots) to illustrate findings.

## Section 4: Timeline
- **Initial Analysis:** 
  - **Time:** 30 min - 2 hours.
  - **Tasks:** Familiarize with data, decide on pre-processing needs.

- **Model Implementation:**
  - **Time:** 3 hours - 5 hours.
  - **Tasks:** Implement and compare ML models.

- **Metric Analysis & Plotting:**
  - **Time:** 2 - 3 hours.
  - **Tasks:** Analyze metrics, create plots.

- **Milestones:**
  1. Data Preprocessing.
  2. ML Model Implementation.
  3. Data Presentation.

## Section 5: Evaluation Plan
- **Evaluation Criteria:**
  - **Data Preprocessing:** Ensure data balance and address duplicates.
  - **ML Models:** Monitor and store metrics (loss, accuracy, precision) to evaluate performance.
  - **Data Presentation:** Adjust plots for interpretability by the target group.

- **Risk Analysis:**
  - **Key Risks:** Lack of significant insights, limited computational resources.
  - **Mitigation:** Simplify models if necessary and manage computational resources effectively.

- **Next Steps:**
  - **Adjustments:** Refine problem statements, goals, requirements, and methodology based on evaluation.