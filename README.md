# Udacity Azure ML Nanodegree Capstone Project - Classifying Malware

Malware is any software designed to intentionally cause harm to a computer system. Viruses, trojans, worms, and ransomware are just a few examples of the types of malicious programs that can find their way on to systems. It is estimated that Wannacry, just one example of ransomware, has already cost over $4 billion and another ransomware attack on a hospital in Germany lead to a woman's death. 

New malware is being created at an alarming rate. Over 400,000 new malwares were identified in 2019 alone and that rate is expected to be higher for 2020. This makes the old methods of signature-based malware detection grossly inefficient. By the time a signature has been created and distributed, the new malware could have already have infected numerous systems. 

To combat this threat, researchers are turning to Machine Learning to help identify malware before it can infect a system. In this project, I will be using a public dataset from Kaggle to train several models to attempt to classify if an exicutible is malware or not. 

## Project Set Up and Installation

To run this project, you will need an active account on Kaggle. From Kaggle, go to your account settings and click the 'Create New API Token' button to download your kaggle.json file. From Azure ML Studio on the Notebooks UI, upload the kaggle.json file and the whole directory from this github repository. Then open a ternimal for the compute instance you will be running the notebooks in. Run the following commands: 

```
cp kaggle.json /home/azureuser/.kaggle/
chmod 600 /home/azureuser/.kaggle/kaggle.json
```

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
