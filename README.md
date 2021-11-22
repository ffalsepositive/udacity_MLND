# Capstone Project - Starbucks Case

## [Proposal](./reports/proposal.pdf)
## [Project Report](./reports/report.pdf)

## Data

- Data sets are provided by Udacity. Three json files (portfolio.json, transcript.json, profile.json) should be stored in <code>./data/raw/</code> directory.

## Project Structure

- data
    - raw
        - portfolio.json
        - profile.json
        - transcript.json <br><br>
         
    - processed
        - train.csv (automatically created by notebook)
        - test.csv (automatically created by notebook) <br><br>
- model 
    - dummy.pkl (automatically created by notebook)
    - logreg.pkl (automatically created by notebook)
    - knn.pkl (automatically created by notebook)
    - dtree.pkl (automatically created by notebook)
    - rforest.pkl (automatically created by notebook)
    - dtree_hyp (automatically created by notebook)
    - rforest_hyp (automatically created by notebook) <br><br>
    
- deployment
    - models
    - app.py
    - process.py
    - templates
        - main.html
        - no_data.html
        - index.html

- starbucks_capstone_data_preparation.ipynb
- starbucks_capstone_model_training.ipynb
- starbucks_capstone_model_evaluation.ipynb

### Required packages

- AWS SageMaker
- JupyterNB
- Python3
    - pandas
    - numpy
    - sklearn
    - matplotlib
    - seaborn
    - pickle
    - flask
    - boto3
    - sagemaker
    
## How to run

Notebooks has been run on AWS SageMaker **ml.t2.medium**
    
    - starbucks_capstone_data_preparation.ipynb
    - starbucks_capstone_model_evaluation.ipynb
    
Notebooks has been run on AWS SageMaker **ml.c4.2xlarge**
    
    - starbucks_capstone_modelling.ipynb
    
Run <code>app.py</code> with the specified environment.
    
_Note:_ S3 Bucket name should be  specified.
