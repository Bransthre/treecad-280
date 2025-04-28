import re
import os
import json
import tqdm

dataset_dir = "../cad-recode-v1.5/train/"
batch_ids = [f"0{i}" for i in range(10)] + list(range(10, 100))


def extract_vocabulary_from_file(file_path, vocabulary_pattern):
    with open(file_path, "r") as file:
        content = file.read()

    # Use regex to find all function calls, the vocab pattern is AI-generated for now
    # Eventually it should be .(anything+)
    function_calls = re.findall(vocabulary_pattern, content)

    # Extract the function names and arguments
    vocabulary = set()
    for call in function_calls:
        # Split by whitespace to separate function name and arguments
        parts = call.split("(")
        if len(parts) > 1:
            func_name = parts[0].strip()
            args = parts[1].rstrip(")").strip()
            vocabulary.add(func_name)

    return vocabulary


all_vocabs = {}
for batch_id in tqdm.tqdm(batch_ids, desc="Processing batches", leave=False):
    batch_dir = os.path.join(dataset_dir, f"batch_{batch_id}")
    vocabulary_pattern = r"\b\w+\s*\(.*?\)"
    all_vocabs_in_batch = {}

    # Check if the directory exists
    if os.path.exists(batch_dir):
        # Iterate through all files in the batch directory
        for _, file_name in tqdm.tqdm(
            enumerate(os.listdir(batch_dir)), desc=f"Processing files in {batch_dir}"
        ):
            file_path = os.path.join(batch_dir, file_name)
            if os.path.isfile(file_path):
                vocabulary = extract_vocabulary_from_file(file_path, vocabulary_pattern)
                all_vocabs_in_batch[file_name] = list(vocabulary)
    else:
        print(f"Directory {batch_dir} does not exist.")

    # Save the vocabulary to a json file
    all_vocabs[batch_id] = all_vocabs_in_batch

output_file = os.path.join("./src", "vocabularies.json")
with open(output_file, "w") as json_file:
    json.dump(all_vocabs, json_file, indent=4)
