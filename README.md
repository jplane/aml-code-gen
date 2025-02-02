# Code Generation Experiments with Azure Machine Learning

This repo demonstrates Python code generation using the [Codestral-2501 model](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/deploy-models-mistral-codestral?pivots=programming-language-python) from Mistral.

It runs multiple code generation test cases in Azure Machine Learning and produces artifacts and [codeBLEU](https://arxiv.org/abs/2009.10297) [scores](https://pypi.org/project/codebleu/) for each test case.

## Setup

- Create an AML workspace
- Provision a Codestral-2501 model endpoint in Azure
- Provision a compute target in AML on which to run the notebook (this minimizes security config issues with running on your laptop)
- Connect to the compute target (browser-based VS Code is a good option) and git pull this repo
- Modify [main.ipynb](./main.ipynb) with your AML workspace and Codestral-2501 model endpoint details
- Run the notebook to create the experiment environment and start a new experiment job (you only need to create the experiment environment once)

## Possible Enhancements

- Make the code gen model configurable and create new experiments with different models
- Add more test cases
- Add more metrics beyond codeBLEU
- Add metrics visualizations (using [matplotlib](https://matplotlib.org/) and mlflow support for storing images, for example)
- Modularize the code... perhaps refactor job stages into [AML components](https://learn.microsoft.com/en-us/azure/machine-learning/concept-component?view=azureml-api-2)
- Configure parallelized job stages
- Trigger automated experiment runs based on updated test cases or experiment code (GH events, etc.)
