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

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
