<img align="left" width="100" height="75" src="https://github.com/dpbac/Optimizing-an-ML-Pipeline-in-Azure/blob/master/images/microsoft-azure-640x401.png">

## Environment Setup

Before we get starting with this second project, it is important to set up our **local development environment** to match with the **Azure AutoML development environment**. 

Below are the steps:

1. Download and install `anaconda`
2. Open `anaconda CMD` in the new folder
3. Clone this repo
4. `cd` into the local directory
5. Run this `conda env create --file udacity_env.yml`
6. Run `jupyter notebook`

## Overview

For this second project of the **Udacity Nanodegree program Machine Learning Engineer with Microsoft Azure** we configure a cloud-based machine learning production model, deploy, 
and consume it.

As in the first project, we use the `Bank Marketing dataset` which contains data collected during direct marketing campaigns (phone calls) of a Portuguese banking institution. 
This is a subset of the original public dataset available at [UCI Machine Learning repository]( https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). In this website a detailed 
description of each feature can be found.

In the [first project]( https://github.com/dpbac/Optimizing-an-ML-Pipeline-in-Azure) the main goal was to optimize an Azure ML pipeline using the Python SDK and a 
provided Scikit-learn model and then compare it to an Azure AutoML run.

In this new project, we go further and not only obtain the best model using Azure Automated ML, but we configure a cloud-based machine learning production model, deploy, 
and consume it.

The main steps performed in this project are:

![Architectural diagram showing main steps](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/main_steps.jpeg)

**Fig. 1 - Architectural diagram showing main steps performed in order to configure a cloud-based machine learning production model, 
deploy it, and consume it.**

source: Adapted from Nanodegree Program Machine Learning Engineer with Microsoft Azure

1. Authentication
2. Automated ML Experiment
3. Deploy the best model
4. Enable logging 
5. Swagger Documentation
6. Consume model endpoints and benchmark endpoints
7. Create and publish a pipeline

## Architectural Diagram

The following image shows the key steps listed above with some more detail.

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/Overview_diagram.png)
**Fig. 2 - Flow of operations of the complete project**


## Key Steps

### Step 1: Authenticatication

It was not necessary to perform this step since this project was developed using the VM provided by Udacity. 
By using the lab provide by Udacity, we are not authorized to create a security principal.

### Step 2: Automated ML Experiment

Having security is enabled and authentication is completed at the first step we create an experiment using `Automated ML`, 
configure a compute cluster, and use that cluster to run the experiment.

In this step we perform the following substeps:

**1. Initialize workspace.**

**2. Create an Azure ML experiment**

**3. Create or Attach an AmlCompute cluster**

**4. Load Dataset from https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv and make sure it is registered.**

**5. Retrieve the best model.**

To start we need to initialize our `workspace` and `create a Azule ML experiment`.
It is important to ensure that the config file is present in the current working directory, i.e., at `.\config.json`. 
`config.json` can be downloaded from home of `Azure Machine Learning Studio`

The `config.json` can be downloaded in the overview of Azure portal. The image below indicates were to find this file.

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/config_json.png)
**Fig. 3 - Where to obtain config.json**

Once you have all settled (Substeps 1 until 3). You are ready to load and register your BankMarketing dataset. The dataset is loaded using this [webaddress](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv).

The following image shows that we successfuly registered our dataset containing the data that will be processed by Azure AutoML.

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/Datastore_Azure.png)
**Fig. 4 - Registered datasets.**

Once the Azure AutoML experiment is complete `(Fig.5)`, we can retrieve the best model `(Fig.6)` to use in the next step.

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/Automl_completed.png)
**Fig. 5 - AutoML experiment completed.**

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/Automl_completed_2.png)
**Fig. 6 - Best model.**

From this last image we can see that the best model is `VotingEnsemble` with **AUC weighted 0.94720**.

### Step 3: Deploy the best model

In this step we select the best model and deploy its enabling authentication using Azure Container Instance (ACI).

To deploy the best model we select `Deploy` in the best run information, give a name to the deployed model, a description, and in our case, choose `Azure Container Instance`. After a little while we can see that our model was successfully deployed! `(Fig.7)`

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/automl_deployed.png)
**Fig. 7 - Model deployed with success.**

Deploying the Best Model will allow to interact with the HTTP API service and interact with the model by sending data over POST requests.

### Step 4: Enable logging

In this step, we work on the [logs.py](https://github.com/franckess/Operationalizing_AzureML/blob/main/logs.py) provided and enable `Applications insights` using Azure SDK. This is a very important step since it allows to determine anomalities, irregularities and visualize the performance. The image below `(Fig. 8)` shows logs.py emphasizing the command that enables Application insights.

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/logs_py.png)

**Fig. 8 - logs.py enabling Applications Insights.**

The next image `(Fig. 9)` shows that `Applicatios Insights` is indeed enabled.

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/application_insight.png)
**Fig. 9 - Applications Insights enabled.**

Since `Applications Insights` is enabled we are able access logs output both at the command line as well as at Endpoints section in Azure ML Studio. 

In addition, we can get insights by checking the performance using the `Applications Insigth url`. You can see in the following image `Fig. 10` that we can, for instances, get information about `Failed requests` and `Server response time`. With this type of information available we can quickly take action if something goes wrong.

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/application_insight_azure.png)
**Fig. 10 - View insights (1/2)**

### Step 5: Swagger Documentation

In this step, we consume the deployed model using `Swagger`. Azure provides a [Swagger](https://swagger.io/) JSON file for deployed models. Consuming the deployed `Endpoints` allows other services to interact with deployed models.

For this we use Swagger, a tool that eases the documentation efforts of HTTP APIs. It helps building document and consuming RESTful web services.

Azure provides swagger.json for deployed models. This file is used to create a web site that documents the HTTP endpoint for a deployed model.

To start we download the swagger json file for the deployed model. It can be found in Section Endpoints in Azure ML Studio. 

**Important**: Make sure that [`swagger.json`](https://github.com/franckess/Operationalizing_AzureML/tree/main/Swagger/swagger.json) is at the same place of [`swagger.sh`](https://github.com/franckess/Operationalizing_AzureML/tree/main/Swagger/swagger.sh) and [`serve.py`](https://github.com/franckess/Operationalizing_AzureML/tree/main/Swagger/serve.py).

Summing up, to consume the deployed model using Swagger you need to:

**1. Download the swagger json file for the deployed model (Section Endpoints)**

**2. Run swagger.sh and serve.py**

The following image we see that Swagger runs on localhost. There we see the HTTP API methods and responses for the model.

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/swagger_automl_deploy.png)
**Fig. 11 - Swagger running on localhost**

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/swagger_automl_score.png)
**Fig. 12 - JSON payload.**

### Step 6: Consume model endpoints and benchmark endpoints

Once the model is deployed, we use `scoring_uri` and `key` in [`endpoint.py`](https://github.com/franckess/Operationalizing_AzureML/blob/main/endpoint.py) script so we can interact with the trained model. 

`endpoint.py` runs against the API producing JSON output from the model ([`data.json`](https://github.com/franckess/Operationalizing_AzureML/blob/main/data.json)).

We also `benchmark` the endpoint using `Apache Benchmark (ab)` running [`benchmark.sh`](https://github.com/franckess/Operationalizing_AzureML/blob/main/benchmark.sh) against the HTTP API using authentication keys to retrieve performance results.

Below we see the output of the Apache Benchmark `(Fig. 13)`.

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/benchmark_part2.png)
**Fig. 13 - Final part of the output when executing benchmark.sh**

Here we can see, for instances, that it took **4.326 seconds** for all the requests to go through and that there were no failed requests. In addition, it takes on average **432 ms per request** which is much less than the limit given by Azure which is 60 seeconds.

### Step 7: Create and publish a pipeline

In this section we show some details about the creation and publishing of a pipeline.

For this in this project we run the [notebook](https://github.com/franckess/Operationalizing_AzureML/blob/main/aml-pipelines-with-automated-machine-learning-step.ipynb) which creates, publishes and consumes a `Azure Machine Learning Pipeline with AutoMLStep`.

The best model is generated using AutoML for classification using the dataset available at https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

The 5 steps necessary to get to our best model are:

**1. Create an `Experiment` in an existing `Workspace`.**

**2. Create or Attach existing AmlCompute to a workspace.**

**3. Define data loading in a `TabularDataset`.**

**4. Configure AutoML using `AutoMLConfig`.**

**5. Use AutoMLStep**

In the figure below we can verify that the pipeline was successfuly created.
<br/><br/>
 
![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/automl_pipeline.png)
**Pipeline created**

![](https://github.com/franckess/Operationalizing_AzureML/blob/main/img/confusion_matrix.png)
**Fig. 23 - Confusion matrix result of testing the pipeline**

Now we reach the point of publishing our pipeline. Publishing a pipeline is the process of making a pipeline publicly available. Here we will use `Python SDK` to publish our pipeline.

When published we can access in the workspace details about the pipeline and we can also run the pipeline manually from the portal.
Additionally, publishing a pipeline, a public HTTP endpoint becomes available, allowing other services, including external ones, to interact with an Azure Pipeline (as seen in the notebook).

## Screen Recording

:movie_camera:     =====>   :man_shrugging:

## Future work

* Try different methods to fight Unbalanced data.

Imbalanced data: 88.80% label `no` and 11.20% label `yes`.

We can apply some techniques on the dataset to make the data more balanced before applying ML. **Automated ML** has built in capabilities to help deal with imbalanced. For instances, the algorithms used by automated ML detect imbalance when the number of samples in the minority class is equal to or fewer than 20% of the number of samples in the majority class, where minority class refers to the one with fewest samples and majority class refers to the one with most samples. Subsequently, AutoML will run an experiment with sub-sampled data to check if using class weights would remedy this problem and improve performance.

For more details, check this [link](https://docs.microsoft.com/en-us/azure/machine-learning/concept-manage-ml-pitfalls).

The following steps were not implemented due to time issues but certainly it would make the project more complete improving performance 
and expanding possibilities of application (e.g. OMNX support)

* Use a [Parallel Run Step](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-run-step) in a pipeline.
* Test a [local container with a downloaded model](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-package-models).
* Export your model to [support ONNX](https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx).



