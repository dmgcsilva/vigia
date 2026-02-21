import json
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import argparse
import os

def calculate_meteor_from_file(file_path):
    """
    Calculates the average METEOR score from a JSON file.

    The JSON file should have a "predictions" field containing a list of dicts.
    Each dict should have a "prediction" and a "target" field.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        float: The average METEOR score.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    total_score = 0
    num_samples = len(data["predictions"])

    if num_samples == 0:
        return 0.0

    for item in data["predictions"]:
        prediction = item["prediction"].replace("**", '').strip().replace("\n", ' ').strip()
        target = item["target"]

        # Tokenize the prediction and target sentences
        tokenized_prediction = word_tokenize(prediction)
        tokenized_target = word_tokenize(target)

        # Calculate the METEOR score
        # The target needs to be a list of reference translations
        score = meteor_score([tokenized_target], tokenized_prediction)
        total_score += score

    return total_score / num_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the METEOR score from a JSON file.")
    parser.add_argument("file_path", type=str, help="The path to the JSON file.")
    args = parser.parse_args()

    average_meteor = calculate_meteor_from_file(args.file_path)
    print(f"Average METEOR score: {average_meteor:.4f}")
    metrics_file = args.file_path.replace('textgen.json', 'metrics.json')
    if os.path.exists(metrics_file):
        print(f"Updating METEOR score in metrics file: {metrics_file}")
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        metrics['textgen']['meteor'] = average_meteor
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)