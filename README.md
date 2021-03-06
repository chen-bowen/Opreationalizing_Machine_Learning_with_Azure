# Opreationalizing Machine Learning with Azure

## Project Overview

In this project, we will use the hybrid of Azure Machine Learning SDK and Azure Machine Learning Studio's webUI. We will start by creating an autoML experiment on the bank marketing prediction dataset and let Azure ML pick the best model for us. Then we will publish the pipeline as an endpoint, which will allow us to trigger the same AutoML experiment through JSON payloads. We will then deploy the best model as a webservice endpoint, add retrieve the endpoint's logging, while also benchmark the endpoint. Finally we will utilize swagger to effectively visualize the API endpoint's request and response for better standardization purposes.

## Architectural Diagram

![Screenshot](images/Architecture.png)

## Key Steps

The general process of operationalizing machine learning on Azure is the following 

1. Create an `Experiment` in an existing `Workspace`.
2. Create or Attach existing AmlCompute to a workspace.
3. Define data loading in a `TabularDataset`.
4. Configure AutoML using `AutoMLConfig`.
5. Configure AutoMLStep
6. Train the model using AmlCompute
7. Explore the results.
8. Test the best fitted model.
9. Publish the pipeline to a REST Endpoint
10. Trigger the pipeline run with a JSON payload
11. Deploy the best performing model from AutoML as a webservice
12. Add logging to the deployed webservice Endpoint
13. Check the Swagger documentation for the webservice endpoint
14. Benchmark the webservice endpoint

(step 1, 2) To get started , create an experiment in the workspace and setting up the compute cluster in the workpace by running the following lines of code for the Azure SDK

```python

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep="\n")


experiment_name = "bank-marketing-prediction-automl"
experiment = Experiment(ws, experiment_name)

# Choose a name for your CPU cluster
amlcompute_cluster_name = "automl-compute"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_D2_V2",  # for GPU, use "STANDARD_NC6" 
        max_nodes=4,
    )
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)
```

We can verify in the workspace the correct resources are created

Experiment (shown as completed)
![Screenshot](images/Completed%20experiment.png)

(step 3) Load in the data by running the following lines from the SDK

```python
found = False
key = "Bank-marketing"
description_text = "Bank Marketing DataSet for Udacity Course 2"

if key in ws.datasets.keys():
    print("Found existing dataset, using")
    found = True
    dataset = ws.datasets[key]

if not found:
    # Create AML Dataset and register it into Workspace
    print(f"Did not find existing dataset with key {key}, creating")
    example_data = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
    dataset = Dataset.Tabular.from_delimited_files(example_data)
    # Register Dataset in Workspace
    dataset = dataset.register(workspace=ws, name=key, description=description_text)
```
This will load in the dataset if it's already in the workspace, otherwise it will create the dataset from that linked datasource
You can go to the `Dataset` tab to verify the dataset you just uploaded is sucessfully registered, a sample screenshot is shown below

![Screenshot](images/Dataset%20verification.png)

(step 4, 5) Create the configurations of the AutoML pipeline. Start by running the following lines in the SDK

```python
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 4,
    "primary_metric": "AUC_weighted",
}
automl_config = AutoMLConfig(
    compute_target=compute_target,
    task="classification",
    training_data=dataset,
    label_column_name="y",
    path=project_folder,
    enable_early_stopping=True,
    featurization="auto",
    debug_log="automl_errors.log",
    model_explainability=True,
    **automl_settings
)

# define outputs
ds = ws.get_default_datastore()
metrics_output_name = "metrics_output"
best_model_output_name = "best_model_output"

metrics_data = PipelineData(
    name="metrics_data",
    datastore=ds,
    pipeline_output_name=metrics_output_name,
    training_output=TrainingOutput(type="Metrics"),
)
model_data = PipelineData(
    name="model_data",
    datastore=ds,
    pipeline_output_name=best_model_output_name,
    training_output=TrainingOutput(type="Model"),
)

# create AutoML step
automl_step = AutoMLStep(
    name="automl_module",
    automl_config=automl_config,
    outputs=[metrics_data, model_data],
    allow_reuse=True,
)

# create pipeline
pipeline = Pipeline(
    description="pipeline_with_automlstep", workspace=ws, steps=[automl_step]
)
```

(step 6) Submit the pipeline for execution 

```python
pipeline_run = experiment.submit(pipeline)
```

You can also find the pipeline you just submitted running in the Azure Machine Learning Studio UI

![Screenshot](images/Completed%20Pipeline.png)

Once the pipeline is completed, you can go to the models tab on the details page, a list of models executed by the AutoML module will be shown, with the best performing model coming up on top. You can click on the name of the model to check on its details.

![Screenshot](images/Completed%20Best%20Model.png)

(step 7, 8) Since you have enabled model explanability, there will be an explanation produced for th best model. Some sample explanations are discussed below.

One interesting thing to look like is the explanation for the best model. The following screenshot shows the importance of each feature

![Screenshot](images/Explanation.png)

Here is another one, this visualization shows the range of the value that could contribute to model predicting a certain class

![Screenshot](images/Explanation_2.png)

(step 9) To publish the pipeline as an endpoint, run following commands 

```python

published_pipeline = pipeline_run.publish_pipeline(
    name="Bankmarketing Train", description="Training bankmarketing pipeline", version="1.0") 
```

You will have to retrieve the authentication header for the pipeline endpoint to be consumeable

```python
interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
```

You can go to the `pipeline endpoints` tab (under `Endpoints` page on your left panel) to verify if your endpoint has been successfully created.

![Screenshot](images/Published%20Pipeline%20Endpoints.png)
 
 If successful, the pipeline status will be shown as active

 ![Screenshot](images/Active%20Pipeline.png)

(step 10) This endpoint could be consumed using JSON payload, which will trigger another scheduled AutoML run

```python
# retrieve the endpoint URI
rest_endpoint = published_pipeline.endpoint
# post a request to the endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": "pipeline-rest-endpoint"}
                        )
```
This will trigger another identical AutoML run, you can quickly verify that by going to the `Experiment` page, a new run will be scheduled, now the title on that experiment run will be called "pipeline rest-endpoint".

![Screenshot](images/Completed%20Pipeline%20endpoint.png)

(step 11) You can also deploy the best model as a web service. This could be easily accomplished by clicking the "Deploy" button 

Once the deployment has been successful, the status on the details page will show the endpoint state as "Healthy"

![Screenshot](images/Deployed%20Best%20Model.png)

At the same time, since you have enabled application insights, the logging URL is also provided

(step 12) You can also mannually trigger the logging to be enabled by running the file `logs.py`. The file will enable logging (from the below line) and print out the logs in the terminal 

```python
service.update(enable_app_insights=True)
```

The logs printed looks like the following

![Screenshot](images/Full%20Logs.png)

Inside the Azure Machine Learning Studio, you will also be able to see a link that provides the usage statics is given to you on the deployed endpoint details page

![Screenshot](images/Application%20Insights%20Status.png)

A sample application insights page is shown below

![Screenshot](images/Application%20Insights.png)

Once deployed, the endpoint could be consumed by sending through some sample payloads. We will create an `endpoint.py` file that contains 2 sample datapoints

```python
data = {
    "data": [
        {
            "age": 17,
            "campaign": 1,
            "cons.conf.idx": -46.2,
            "cons.price.idx": 92.893,
            "contact": "cellular",
            "day_of_week": "mon",
            "default": "no",
            "duration": 971,
            "education": "university.degree",
            "emp.var.rate": -1.8,
            "euribor3m": 1.299,
            "housing": "yes",
            "job": "blue-collar",
            "loan": "yes",
            "marital": "married",
            "month": "may",
            "nr.employed": 5099.1,
            "pdays": 999,
            "poutcome": "failure",
            "previous": 1,
        },
        {
            "age": 87,
            "campaign": 1,
            "cons.conf.idx": -46.2,
            "cons.price.idx": 92.893,
            "contact": "cellular",
            "day_of_week": "mon",
            "default": "no",
            "duration": 471,
            "education": "university.degree",
            "emp.var.rate": -1.8,
            "euribor3m": 1.299,
            "housing": "yes",
            "job": "blue-collar",
            "loan": "yes",
            "marital": "married",
            "month": "may",
            "nr.employed": 5099.1,
            "pdays": 999,
            "poutcome": "failure",
            "previous": 1,
        },
    ]
}
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {"Content-Type": "application/json"}
# If authentication is enabled, set the authorization header
headers["Authorization"] = f"Bearer {key}"

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
```

If the endpoint is fully functional, the response will directly show whether the customer is likely to subscribe a term deposit in the future


![Screenshot](images/endpoint%20response.png)

(step 13) Swagger provided a simple way to understand the request and response format of the API. Azure ML provided a `swagger.json` file that allow us to run a swagger server in your local environment. On two separate terminal windows, you can run the `serve.py` and `swagger.sh` file, which allow us access swagger with our local host. You can then type in "https://localhost" into your browser, then swagger server will be loaded up. Then you will replace the default url with `http://localhost:8000/swagger.json` to view the API's documentation

The swagger server looks like the following,

![Screenshot](images/swagger.png)

Our API's request and response format looks like the following

Request

![Screenshot](images/swagger_2.png)

Response

![Screenshot](images/swagger_3.png)

(Step 14 optional) You can use apache's benchmark functionality to view if our API is having any irregularities

Type in the following command in the terminal

``` ab -n 10 -v 4 -p data.json -T 'application/json' -H 'Authorization: Bearer PRIMARY_KEY' SCORE_URI```

(remember to replace key and scoring URI)

Then the benchmark stats will shown in the terminal which looks like the following

![Screenshot](images/benchmarking.png)

All of the steps could be viewd in the jupyter notebook.

## Screen Recording
You can watch a screencast demo from the link [here](https://youtu.be/nx3_LPhnGgI) ~ 5 min

## Future Improvements

* **Data balancing** - Dataset currently has a strong bias towards 0 class, AutoML has already alerted us on that. Some data balancing techniques (like over/under sampling) could be employed to tackle this issue

* **Automate deployment of best model** - Right now, with the pipeline rest endpoint we are able to kick off automatic autoML runs, but there's no easy way to automate the process of automatically deploy the new best model. Some more extensive work on the Azure SDK could potentially solve this issue

* **Full control on deployment environment** - Azure Container Instance (ACI) offers an easy way for deployments, but lack of configuration customization capablities. We could potentially try to deploy the model to Azure Kuberneted Instance (AKI) to get full control over the environment the model is running in

## Standout Suggestions
* getting the terminal commands into the jupyter notebook for better visualization purposes. 
* obtain sensitive information (API URI and primary key) dynamically from code
* Attempt to deploy the best model to ACI (wasn't successful)