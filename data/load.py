import torch
import json
import os

def load(file_path: str, dataset):
    for json_file in os.listdir(file_path):
        if json_file.endswith(".json"):
            with open(os.path.join(file_path, json_file), "r") as f:
                data = json.load(f)
            
                data["P"] = torch.tensor(data["P"])

                dataset.add_element(data)