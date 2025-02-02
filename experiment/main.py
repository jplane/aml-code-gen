import os
import argparse
import pandas as pd
import mlflow
from codebleu import calc_codebleu
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_cases_path", type=str)
    args = parser.parse_args()

    llm_name = os.environ['LLM_NAME']
    code_gen_iterations = int(os.environ['CODE_GEN_ITERATIONS'])

    mlflow.start_run()

    tags = {
        'llm_name': llm_name
    }

    mlflow.set_tags(tags)

    mlflow.log_param('code_gen_iterations', code_gen_iterations)

    with open('system_prompt.txt', 'r') as system_prompt_file:
        system_prompt = system_prompt_file.read()

    results = pd.DataFrame(columns=['test_name', 'test_iteration', 'response_tokens', 'codebleu', 'ngram_match_score', 'weighted_ngram_match_score', 'syntax_match_score', 'dataflow_match_score'])

    for subdir, _, _ in os.walk(args.test_cases_path):

        test_name = os.path.basename(subdir)
        prompt_path = os.path.join(subdir, 'prompt.txt')
        expected_path = os.path.join(subdir, 'expected.py')

        with open(prompt_path, 'r') as prompt_file:
            user_prompt = prompt_file.read()

        with open(expected_path, 'r') as expected_file:
            expected_code = expected_file.read()

        for iteration in range(code_gen_iterations):

            actual_code, response_tokens = generate_code(system_prompt, user_prompt)

            with open('actual.py', 'w') as actual_file:
                actual_file.write(actual_code)
                mlflow.log_artifact(os.path.join(test_name, f'actual_{iteration}.py'))

            scores = calc_codebleu([expected_code], [actual_code], lang='python')

            scores['test_name'] = test_name
            scores['test_iteration'] = iteration
            scores['response_tokens'] = response_tokens

            results = results.append(scores, ignore_index=True)

    store_results(results)

    mlflow.end_run()


def store_results(results):

    mean_response_tokens = results['response_tokens'].mean()
    mlflow.log_metric('mean_response_tokens', mean_response_tokens)

    mean_codebleu = results['codebleu'].mean()
    mlflow.log_metric('mean_codebleu', mean_codebleu)

    mean_ngram_match_score = results['ngram_match_score'].mean()
    mlflow.log_metric('mean_ngram_match_score', mean_ngram_match_score)

    mean_weighted_ngram_match_score = results['weighted_ngram_match_score'].mean()
    mlflow.log_metric('mean_weighted_ngram_match_score', mean_weighted_ngram_match_score)

    mean_syntax_match_score = results['syntax_match_score'].mean()
    mlflow.log_metric('mean_syntax_match_score', mean_syntax_match_score)

    mean_dataflow_match_score = results['dataflow_match_score'].mean()
    mlflow.log_metric('mean_dataflow_match_score', mean_dataflow_match_score)

    results.to_csv('results.csv', index=False)
    mlflow.log_artifact('results.csv')


def generate_code(system_prompt, user_prompt):

    llm_endpoint_path = os.environ['LLM_ENDPOINT_PATH']
    llm_cred = os.environ['LLM_CREDENTIAL']
    llm_temp = int(os.environ['LLM_TEMP'])
    llm_top_p = int(os.environ['LLM_TOP_P'])

    mlflow.log_param('llm_temp', llm_temp)
    mlflow.log_param('llm_top_p', llm_top_p)

    client = ChatCompletionsClient(endpoint=llm_endpoint_path, credential=AzureKeyCredential(llm_cred))

    response = client.complete(
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt),
        ],
        temperature=llm_temp,
        top_p=llm_top_p
    )

    return response.choices[0].message.content, response.usage.completion_tokens


if __name__ == "__main__":
    main()
