"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 4/6/2021

Purpose:

Details:

Description:

Notes:

IMPORTANT NOTES:

Explanation:

Reference:
    SQUAD dataset | Preprocessing and Download
        Reference:
            https://www.youtube.com/watch?v=tsc6ALuKJVk
"""
import json
import os
from typing import Generator

import pandas as pd

print(os.getcwd())

# Datasets are not sanitized, uise with open
FILE_DATASET_TRAIN = os.path.join("H:/", "Datasets", "SQuAD_2", "train-v2.0.json")
FILE_DATASET_TEST = os.path.join("H:/", "Datasets", "SQuAD_2", "dev-v2.0.json")


# print(json.loads(FILE_DATASET_TRAIN))
# with open(FILE_DATASET_TRAIN, encoding="UTF-8") as f:
#     print(json.load(f))

def _test():
    pd_df_sentences = pd.read_json(FILE_DATASET_TRAIN)
    print(pd_df_sentences[0:-1])


def _get_text(json_dataset) -> Generator:
    """
    Load the text from SQuAD dataset

    :return:
    """
    with open(json_dataset, encoding="UTF-8") as f:
        file_json = json.load(f)

    for i in file_json["data"]:
        for j in i["paragraphs"]:
            for k in j["qas"]:
                for a in k["answers"]:
                    text = a["text"]
                    yield text
                    # print(text)
            context = j["context"]
            # print(context)
            yield context


def get_gen_text_train() -> Generator:
    return _get_text(FILE_DATASET_TRAIN)


def get_gen_text_test() -> Generator:
    return _get_text(FILE_DATASET_TEST)


if __name__ == '__main__':
    print(get_gen_text_train())
    print(get_gen_text_test())
