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
The Portable Executable (PE) format is a file format for executables, object code, DLLs and others used in 32-bit and 64-bit versions of Windows operating systems. The header of PE files contains a number for things like the size of the file, imported libraries, and more. This dataset from [Kaggle](https://www.kaggle.com/divg07/malware-analysis-dataset) contains data extracted from PE headers from both known malware samples and benign software samples.

### Task
The task for this project is to train models to classify whether an executable is malware or benign using features extracted from their PE Header. The 'legitimate' column in the dataset is 1 when the executible file is from a legitimate source (aka benign software or goodware), and 0 when it is malware.  

### Access
We will be downloading the data from Kaggle directly using the kaggle python api. See the Project Set Up and Installation section above to make sure you have the kaggle.json in the correct location to use this api. Once the data is downloaded, there is some minor data cleaning before the dataset is registered to the Azure ML workspace. 

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
The [automl](https://github.com/DrewAumick/nd00333-capstone/blob/master/automl.ipynb) notebook will run you through the steps of configuring and running the AutoML experiment. We do a Classification task on the 'legitimate' column from the malware dataset. We also set the primary metric to 'accuracy' with auto featurization, and a timeout set at 30 minutes.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
For the Hyperdrive run, I wrote a custom training script using a RandomForestClassifier from Scikit Learn. Random Forest models generally provide a high accuracy in a relatively short training time. For the hyperparameter tuning of this model, we will be tuning four different paramaters for the forest using a random parameter sampling:
* n_estimators: The number of trees in the Random forrest - choice of 10, 50, 100, 150, 200
* max_depth: The maximum depth of the trees in the forrest - choice of 0, 2, 5, 10
* min_samples_split: The minimum number of samples required to split an internal node - choice of 2, 3, 4, 5
* min_samples_leaf: The minimum number of samples required to be at a leaf node - choice of 1, 2, 3, 4, 5

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
