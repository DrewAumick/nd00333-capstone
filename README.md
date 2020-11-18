# Udacity Azure ML Nanodegree Capstone Project - Classifying Malware

Malware is any software designed to intentionally cause harm to a computer system. Viruses, trojans, worms, and ransomware are just a few examples of the types of malicious programs that can infect systems. It is estimated that Wannacry, just one example of ransomware, has already cost over $4 billion and another ransomware attack on a hospital in Germany lead to a woman's death. 

New malware is being created at an alarming rate. Over 400,000 new malwares were identified in 2019 alone and that rate is expected to be higher for 2020. This makes the old methods of signature-based malware detection grossly inefficient. By the time a signature has been created and distributed to anti-virus software, the new malware could have already have infected numerous systems. 

To combat this threat, researchers are turning to Machine Learning to help identify malware before it can infect a system. In this project, I will be using a public dataset from Kaggle to train several models to attempt to classify if an exicutible is malware or not. 

## Project Set Up and Installation
To run this project, you will need an active account on Kaggle. From Kaggle, go to your account settings and click the 'Create New API Token' button to download your kaggle.json file. From Azure ML Studio on the Notebooks UI, upload the kaggle.json file and the whole directory from this github repository. Then open a ternimal for the compute instance you will be running the notebooks in. Run the following commands: 

```
pip install kaggle
cp kaggle.json /home/azureuser/.kaggle/
chmod 600 /home/azureuser/.kaggle/kaggle.json
```

## Dataset

### Overview
The Portable Executable (PE) format is a file format for executables, object code, DLLs and others used in 32-bit and 64-bit versions of Windows operating systems. The header of PE files contains a number for things like the size of the file, imported libraries, and more. This dataset from [Kaggle](https://www.kaggle.com/divg07/malware-analysis-dataset) contains data extracted from PE headers from over 130,000 executible files, about half are known malware samples and the other half are benign software samples.

### Task
The task for this project is to train models to classify whether an executable is malware or benign using features extracted from their PE Header. The 'legitimate' column in the dataset is 1 when the executible file is from a legitimate source (aka benign software or goodware), and 0 when it is malware.  

### Access
We will be downloading the data from Kaggle directly using the Kaggle Python API. See the Project Set Up and Installation section above to make sure you have the kaggle.json file in the correct location to use this API. Once the data is downloaded, there is some minor data cleaning before the dataset is registered to the Azure ML workspace. 

## Automated ML
The [automl](https://github.com/DrewAumick/nd00333-capstone/blob/master/automl.ipynb) notebook will run you through the steps of configuring and running the AutoML experiment. We do a Classification task on the 'legitimate' column from the malware dataset. We also set the primary metric to 'accuracy' with auto featurization, and a timeout set at 30 minutes.

### Results
Below we see the results from the AutoML run:
![AutoML RunDetails](https://github.com/DrewAumick/nd00333-capstone/blob/master/Udacity%20Capstone%20Screenshots/automl%20rundetails.PNG)
![AutoML Run](https://github.com/DrewAumick/nd00333-capstone/blob/master/Udacity%20Capstone%20Screenshots/automl%20complete%20run.PNG)

The best run was a VotingEnsemble model, which collects the weighted results from several other models (in this case several LightGBM and XGBoost Classifier models). This was able to achieve a 99.996% accuracy, which, if this could be efficiently applied to real malware in the wild, could be incredible for keeping people's computers safe.

![Best AutoML Model UI](https://github.com/DrewAumick/nd00333-capstone/blob/master/Udacity%20Capstone%20Screenshots/automl%20best%20model%20ui.PNG)
![Best AutoML Model Notebook](https://github.com/DrewAumick/nd00333-capstone/blob/master/Udacity%20Capstone%20Screenshots/automl%20best%20model%20notebook.PNG)

## Hyperparameter Tuning
The [hyperparamter tuning](https://github.com/DrewAumick/nd00333-capstone/blob/master/hyperparameter_tuning.ipynb) notebook will run you through the steps for the Hyperdrive run. I wrote a custom training script using a RandomForestClassifier from Scikit Learn. Random Forest models generally provide a high accuracy in a relatively short training time. For the hyperparameter tuning of this model, we will be tuning four different paramaters for the forest using a random parameter sampling:
* n_estimators: The number of trees in the Random forrest - choice of 10, 50, 100, 150, or 200
* max_depth: The maximum depth of the trees in the forrest - choice of 2, 5, 10, or 0 (i.e. no maximum depth, so the tree will keep growing until the model decides to stop it)
* min_samples_split: The minimum number of samples required to split an internal node - choice of 2, 3, 4, or 5
* min_samples_leaf: The minimum number of samples required to be at a leaf node - choice of 1, 2, 3, 4, or 5

### Results
The Random Forest model also did very well at classifying this dataset. 

![HyperDrive RunDetails](https://github.com/DrewAumick/nd00333-capstone/blob/master/Udacity%20Capstone%20Screenshots/hyperdrive%20rundetails.PNG)
![HyperDrive Run UI](https://github.com/DrewAumick/nd00333-capstone/blob/master/Udacity%20Capstone%20Screenshots/hyperdrive%20completed%20run.PNG)

We can see here that, in general, setting the Max Depth of the tree to 0 (i.e. letting the model itself decide how deep the tree should be) did better than speficfying the max depth. It also seems like having more trees in the forest improved the accuracy, which also seems intuitive. Our best model for HyperDrive had an accuracry of 99.94%, which is still incredibly high, but a little shy of the AutoML run.

You can see the exact parameters and some more details for the best run in the screenshots below:

![Best Hyperdrive Run](https://github.com/DrewAumick/nd00333-capstone/blob/master/Udacity%20Capstone%20Screenshots/hyperdrive%20best%20run.PNG)
![Best Hyperdrive Run UI](https://github.com/DrewAumick/nd00333-capstone/blob/master/Udacity%20Capstone%20Screenshots/hyperdrive%20best%20run%20ui.PNG)

## Model Deployment
Just edging out the HyperDrive best model, the AutoML Voting Ensemble model is the one that I deployed to a REST endpoint for inference. By using the AutoML generated model, you can use the scoring script and environment it creates itself and saves in it's output folder for the inference configuration. 

In this screenshot, we can see the model and see that it's been registered as 'best_automl_model' and has it's service created as 'best-model-service':
![AutoML Best Model UI](https://github.com/DrewAumick/nd00333-capstone/blob/master/Udacity%20Capstone%20Screenshots/automl%20best%20model%20ui.PNG)

And in this screenshot, we can see the details of the endpoint that was created, including the URL for the REST endpoint:
![Model Endpoint](https://github.com/DrewAumick/nd00333-capstone/blob/master/Udacity%20Capstone%20Screenshots/model%20endpoint.PNG)

To submit to this endpoint, you need JSON in the following format:
```
{"data": 
  [
    {
    "Name": 101970, 
    "md5": 36283, 
    "Machine": 332, 
    "SizeOfOptionalHeader": 224, 
    "Characteristics": 8450, 
    "MajorLinkerVersion": 9, 
    "MinorLinkerVersion": 0, 
    "SizeOfCode": 512, 
    "SizeOfInitializedData": 1536, 
    "SizeOfUninitializedData": 0, 
    "AddressOfEntryPoint": 4205, 
    "BaseOfCode": 4096, 
    "BaseOfData": 8192, 
    "ImageBase": 268435456.0, 
    "SectionAlignment": 4096, 
    "FileAlignment": 512, 
    "MajorOperatingSystemVersion": 6, 
    "MinorOperatingSystemVersion": 1, 
    "MajorImageVersion": 6, 
    "MinorImageVersion": 1, 
    "MajorSubsystemVersion": 6, 
    "MinorSubsystemVersion": 1, 
    "SizeOfImage": 16384, 
    "SizeOfHeaders": 1024, 
    "CheckSum": 58572, 
    "Subsystem": 3, 
    "DllCharacteristics": 1344, 
    "SizeOfStackReserve": 262144, 
    "SizeOfStackCommit": 4096, 
    "SizeOfHeapReserve": 1048576, 
    "SizeOfHeapCommit": 4096, 
    "LoaderFlags": 0, 
    "NumberOfRvaAndSizes": 16, 
    "SectionsNb": 3, 
    "SectionsMeanEntropy": 2.3739614670099995, 
    "SectionsMinEntropy": 0.020393135236099997, 
    "SectionsMaxEntropy": 3.7514938597900005, 
    "SectionsMeanRawsize": 682.666666667, 
    "SectionsMinRawsize": 512, 
    "SectionMaxRawsize": 1024, 
    "SectionsMeanVirtualsize": 455.0, 
    "SectionsMinVirtualsize": 26, 
    "SectionMaxVirtualsize": 1000, 
    "ImportsNbDLL": 0, 
    "ImportsNb": 0, 
    "ImportsNbOrdinal": 0, 
    "ExportNb": 5, 
    "ResourcesNb": 1,
    "ResourcesMeanEntropy": 3.56688013997, 
    "ResourcesMinEntropy": 3.56688013997, 
    "ResourcesMaxEntropy": 3.56688013997,
    "ResourcesMeanSize": 900.0, 
    "ResourcesMinSize": 900, 
    "ResourcesMaxSize": 900,
    "LoadConfigurationSize": 0, 
    "VersionInformationSize": 16
    }
   ]
}
```
## Screen Recording
Click [here](https://youtu.be/GrIkKLyh0lE) to find a screencast recording of me going through this project and running some sample data against the deployed endpoint.

## Future Work
In the future, doing some some more featurization or data cleaning on the dataset before feeding them into either method might be able to improve results. However, with accuracies already in the 99% range, it seems like it could be difficult to improve much and may not be worth the time. 

More study also needs to be done on different families of Malware. It's unclear at a glance what exact types of malware were used in this dataset, so while it does very well on certain types presented in the data, it may not do as well on truly novel malware. 

I also chose this dataset in part because it was small enough for the Udacity lab to handle downloading and training on the data with the limited cpu and memory quotas and within the 4 hours we had before the lab expired. I initially tired a few other datasets from kaggle that proved too large for this lab to handle with the given resources. In the future, I would try similar experiments on more robust Azure VMs with larger datasets that would hopefully contain even more types of malware.   

