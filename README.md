<img align="left" width="100" height="75" src="https://github.com/dpbac/Optimizing-an-ML-Pipeline-in-Azure/blob/master/images/microsoft-azure-640x401.png">

## Overview

In this second project of the **Udacity Nanodegree program Machine Learning Engineer with Microsoft Azure** we configure a cloud-based machine learning production model, deploy, 
and consume it.

As in the first project, we use the `Bank Marketing dataset` which contains data collected during direct marketing campaigns (phone calls) of a Portuguese banking institution. 
This is a subset of the original public dataset available at [UCI Machine Learning repository]( https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). In this website a detailed 
description of each feature can be found.

In the [first project]( https://github.com/dpbac/Optimizing-an-ML-Pipeline-in-Azure) the main goal was to optimize an Azure ML pipeline using the Python SDK and a 
provided Scikit-learn model and then compare it to an Azure AutoML run.

In this new project, we go further and not only obtain the best model using Azure Automated ML, but we configure a cloud-based machine learning production model, deploy, 
and consume it.

The main steps performed in this project are:

![Architectural diagram showing main steps](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/main_steps.JPG)

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

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/architectural_diagram.JPG)
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

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/configure_json_edited.JPG)
**Fig. 3 - Where to obtain config.json**

Once you have all settled (Substeps 1 until 3). You are ready to load and register your BankMarketing dataset. The dataset is loaded using this [webaddress](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv).

The following image shows that we successfuly registered our dataset containing the data that will be processed by Azure AutoML.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/registered_datasets.JPG)
**Fig. 4 - Registered datasets.**

Once the Azure AutoML experiment is complete `(Fig.5)`, we can retrieve the best model `(Fig.6)` to use in the next step.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/experiment_completed.JPG)
**Fig. 5 - AutoML experiment completed.**

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/best_model.JPG)
**Fig. 6 - Best model.**

From this last image we can see that the best model is `Voting Ensemble` with AUC weighted 0.9463.

### Step 3: Deploy the best model

In this step we select the best model and deploy its enabling authentication using Azure Container Instance (ACI).

To deploy the best model we select `Deploy` in the best run information, give a name to the deployed model, a description, and in our case, choose `Azure Container Instance` as show in the image below `(Fig.7)`.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/deploy_model.JPG)
**Fig. 7 - Deploying the best model.**

After a little while we can see that our model was successfully deployed! `(Fig.8)`

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/model_deploy_succeed.JPG)
**Fig. 8 - Model deployed with success.**

Deploying the Best Model will allow to interact with the HTTP API service and interact with the model by sending data over POST requests.

### Step 4: Enable logging

In this step, we work on the [logs.py](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/logs.py) provided and enable `Applications insights` using Azure SDK. This is a very important step since it allows to determine anomalities, irregularities and visualize the performance. The image below `(Fig. 9)` shows logs.py emphasizing the command that enables Application insights.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/logspy2.JPG)

**Fig. 9 - logs.py enabling Applications Insights.**

The next image `(Fig. 10)` shows that `Applicatios Insights` is indeed enabled.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/application_insights_enabled.JPG)
**Fig. 10 - Applications Insights enabled.**

Since `Applications Insights` is enabled we are able access logs output both at the command line as well as at Endpoints section in Azure ML Studio. `Fig.11` shows logs at the command line, while `Fig. 12` shows logs at Azure ML Studio.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/logs_command_line.png)

**Fig. 11 - Example output logs.py.**

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/log01.JPG)
**Fig. 12 - Output logs at Azure ML Studio.**

In addition, we can get insights by checking the performance using the `Applications Insigth url`. You can see in the following images `(Figs. 13 and 14)` that we can, for instances, get information about `Failed requests` and `Server response time`. With this type of information available we can quickly take action if something goes wrong.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/insights_02.JPG)
**Fig. 13 - View insights (1/2)**

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/insights_03.JPG)
**Fig. 14 - View insights (2/2)**

### Step 5: Swagger Documentation

In this step, we consume the deployed model using `Swagger`. Azure provides a [Swagger](https://swagger.io/) JSON file for deployed models. Consuming the deployed `Endpoints` allows other services to interact with deployed models.

For this we use Swagger, a tool that eases the documentation efforts of HTTP APIs. It helps building document and consuming RESTful web services.

Azure provides swagger.json for deployed models. This file is used to create a web site that documents the HTTP endpoint for a deployed model.

To start we download the swagger json file for the deployed model. It can be found in Section Endpoints in Azure ML Studio. 

**Important**: Make sure that [`swagger.json`](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/swagger/swagger.json) is at the same place of [`swagger.sh`](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/swagger/swagger.sh) and [`serve.py`](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/swagger/serve.py).

Summing up, to consume the deployed model using Swagger you need to:

**1. Download the swagger json file for the deployed model (Section Endpoints)**

**2. Run swagger.sh and serve.py**

The following image we see that Swagger runs on localhost. There we see the HTTP API methods and responses for the model.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/localhost_json.JPG)
**Fig. 15 - Swagger running on localhost**

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/localhost_swagger03.JPG)
**Fig. 16 - JSON payload.**

### Step 6: Consume model endpoints and benchmark endpoints

Once the model is deployed, we use `scoring_uri` and `key` in [`endpoint.py`](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/endpoint.py) script so we can interact with the trained model. 

`endpoint.py` runs against the API producing JSON output from the model ([`data.json`](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/data.json)).

We also `benchmark` the endpoint using `Apache Benchmark (ab)` running [`benchmark.sh`](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/benchmark.sh) against the HTTP API using authentication keys to retrieve performance results.

The following image `(Fig.17)` shows `endpoint.py` script runs against the API producing JSON output from the model, as well as `benchmark.sh` runs.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/endpoint_datajson_benchmark.JPG)
**Fig. 17 - Output endpoint.py, data.json (obtained when running endpoint.py), and output of benchmark.sh.**

Next we see the output of the Apache Benchmark `(Fig. 18)`.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/endpoint_datajson_benchmark_04.JPG)
**Fig. 18 - Final part of the output when executing benchmark.sh**

Here we can see, for instances, that it took **1.540 seconds** for all the requests to go through and that there were no failed requests. In addition, it takes on average **154 ms per request** which is much less than the limit given by Azure which is 60 seeconds.

### Step 7: Create and publish a pipeline

In this section we show some details about the creation and publishing of a pipeline.

For this in this project we run the [notebook](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/auto-pipelines-with-ml-classification-bank-marketing_01012021_run.ipynb) which creates, publishes and consumes a `Azure Machine Learning Pipeline with AutoMLStep`.

The best model is generated using AutoML for classification using the dataset available at https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

As explained at the begining of this README 5 steps are necessary until we get to our best model. Therefore, in this 
notebook we start performing those steps (1-4) in addition to step 5 that is the responsable of creating the pipeline:

**1. Create an `Experiment` in an existing `Workspace`.**

**2. Create or Attach existing AmlCompute to a workspace.**

**3. Define data loading in a `TabularDataset`.**

**4. Configure AutoML using `AutoMLConfig`.**

**5. Use AutoMLStep**

In `Fig. 19` we can verify that the pipeline was successfuly created.
<br/><br/>
 
![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/pipelines_runs.JPG)
**Fig. 19 - Pipeline created**

The pipeline includes all previous steps so we can see again the Bankmarketing dataset with the AutoML module `(Fig. 20)`.
<br/><br/>

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/registered_datasets.JPG)
**Fig. 20 - Registered datasets.**

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/details_dataset.JPG)
**Fig. 21 - Details dataset**

Now we can train the model using `AmlCompute`. After training the model we can observe that the run was completed with the help of `Use RunDetails Widget` `(Fig. 22)`. 
<br/><br/>
![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/run_details_widget.JPG)
**Fig. 22 - Run details**

Outputs of above run can be used as inputs of other steps in pipeline. 

Now we retrieve metrics and the best model.Then we test the pipeline. To see all metrics of all runs please check cell 2 of the 
[notebook](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/auto-pipelines-with-ml-classification-bank-marketing.ipynb).

Then we can use a test dataset to test our pipeline. The image below shows the confusion matrix obtained from our test.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/confusion_matrix_test.JPG)
**Fig. 23 - Confusion matrix result of testing the pipeline**

Now we reach the point of publishing our pipeline. Publishing a pipeline is the process of making a pipeline publicly available. Here we will use `Python SDK` to publish our pipeline.

When published we can access in the workspace details about the pipeline and we can also run the pipeline manually from the portal.
Additionally, publishing a pipeline, a public HTTP endpoint becomes available, allowing other services, including external ones, to interact with an Azure Pipeline.

At `Published Pipeline overview`, we can see a REST endpoint and a status of ACTIVE `(Fig. 24)`.

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/published_pipeline_active.JPG)

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/published_pipeline_overview_RESTPOINT_ACTIVE.JPG)

**Fig. 24 -Published pipeline active**

In the last two images we can see, respectively, that our `pipeline endpoint` is active and details about the `pipeline endpoint` runs.


![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/pipelines_endpoints.JPG)

**Fig. 25 -Pipelines endpoints**

![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/published_pipeline_schedule_run.JPG)

**Fig. 26 -Schedule run**

<!-- ![](https://github.com/dpbac/Operationalizing-Machine-Learning-with-Azure/blob/master/images/run_details_widget.JPG)

**Fig. 27 -Run details of published pipeline**

 -->
## Screen Recording

:movie_camera: https://youtu.be/QMO7mhYaGGw

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



