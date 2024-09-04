# AI in Education Hackathon 2024

## Overview
This repository contains the project files and documentation for the "AI in Education" seminar at LUH, Summer Semester 2024. The project aims to investigate biases in student performance based on their ethnicity, gender, and parental educational background using machine learning techniques. This analysis will help us understand potential inequalities in educational opportunities and propose ways to create fairer educational environments.

## Seminar Information
- **Date/Time**: 3-4 September 2024 (both days 9.15-17.30)

## Project Objectives
The primary goal of this project is to analyze and understand the impact of different socio-economic factors (like ethnicity, gender, and parental education) on student performance. By identifying patterns in the data, we aim to detect potential biases and provide recommendations to promote fairness in educational settings.

### Key Questions
1. Are there disparities in performance between different demographic groups?
2. Is there evidence suggesting unfair chances based on ethnicity, gender, or parental education?
3. How can educators use these findings to create fairer educational opportunities?

## Methodology
1. **Data Analysis**: Use K-Fold Cross-Validation to split the dataset into training, validation, and testing subsets.
2. **Model Development**: Experiment with different machine learning models (e.g., Random Forests, Decision Trees, Logistic Regression) to find the best fit for detecting bias in the dataset.
3. **Evaluation**: Assess models using metrics such as accuracy, precision, ROC-AUC, and cross-validation to ensure reliability.
4. **Visualization**: Generate interpretable plots to present the findings in an accessible way.

## Project Phases
1. **Phase 1**: Initial Data Analysis - Understand and preprocess the dataset.
2. **Phase 2**: Model Development - Develop and test different ML models to find the most effective one.
3. **Phase 3**: Visualization - Generate plots to visualize key findings.
4. **Phase 4**: Presentation - Present the findings, conclusions, and potential recommendations.

## Deliverables
- **Project Plan**: A structured document outlining the problem definition, goals, methodology, timeline, and evaluation criteria.
- **Prototype**: An initial implementation of the chosen model with supporting scripts for data preprocessing, analysis, and visualization.
- **Presentation**: A concise presentation summarizing our approach, findings, and lessons learned.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Usage
To run the project locally, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/detixdeti/AI-In-Education-Hackathon-2024.git
cd AI-In-Education-Hackathon-2024
pip install -r requirements.txt
