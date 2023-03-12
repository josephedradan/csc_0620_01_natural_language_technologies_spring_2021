"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 4/7/2021

Purpose:

Details:

Description:

Notes:

IMPORTANT NOTES:

Explanation:

Reference:
    Pretrained models
        Notes:
            All base pretrained models

        Reference:
            Pretrained models

    AlbertForQuestionAnswering
        Notes:
            What I used as reference

        Reference:
            https://huggingface.co/transformers/model_doc/albert.html#albertforquestionanswering

    SQuAD leader boards
        Notes:
            I used it to choose a good model_bart

        Reference:
            https://rajpurkar.github.io/SQuAD-explorer/

"""
from typing import Tuple, List

import torch
from joseph_library.decorators import timer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AlbertForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.auto.tokenization_auto import AlbertTokenizerFast

"""
Base models from huggingface

Reference:
    Pretrained models
        Reference:
            https://huggingface.co/transformers/pretrained_models.html

"""
MODEL_PRETRAINED_DISTILBERT_BASE_UNCASED = 'distilbert-base-uncased'
MODEL_PRETRAINED_DISTILBERT_BASE_UNCASED_DISTILLED_SQUAD = 'distilbert-base-uncased-distilled-squad'

"""

Reference:
    bert-large-uncased-whole-word-masking-finetuned-squad
        Notes:
            Bert large fine tuned on SQuAD 1.0
        Reference:
            https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad

"""
MODEL_PRETRAINED_BERT_LARGE_UNCASED_WHOLE_WORD_MASKING_FINETUNED_SQUAD = "bert-large-uncased-whole-word-masking-finetuned-squad"

"""

Reference:
    ktrapeznikov/albert-xlarge-v2-squad-v2
        Notes:
            User uploaded pretrained albert xlarge v2 fined tuned on SQuAD 2.0
        Reference:
            https://huggingface.co/ktrapeznikov/albert-xlarge-v2-squad-v2
"""
MODEL_PRETRAINED_ALBERT_XLARGE_V2_SQUAD_V2 = "ktrapeznikov/albert-xlarge-v2-squad-v2"


@timer
class QuestionAndAnsweringSystemHandler:

    def __init__(self, name_model_huggingface: str):
        """
        Automatically assign the model_question_answer and the tokenizer based on the model_question_answer given

        :param name_model_huggingface:
        """

        # Model for question and answering
        self.model_question_answer: AlbertForQuestionAnswering = AutoModelForQuestionAnswering.from_pretrained(
            name_model_huggingface)  # THE CASTING FOR THE MODEL IS FOR AUTO SUGGESTING

        # Tokenizer associated with model_bart
        self.tokenizer: AlbertTokenizerFast = AutoTokenizer.from_pretrained(
            name_model_huggingface)  # THE CASTING FOR THE MODEL IS FOR AUTO SUGGESTING

    # @timer
    def answer_question(self, str_question: str, str_context: str) -> Tuple[str,
                                                                            List[str],
                                                                            List[int],
                                                                            List[str],
                                                                            QuestionAnsweringModelOutput]:
        """
        Answer question given tuple_given

        References:
            Extractive Question Answering
                Notes:
                    The example given is using huggingface 3, must convert to version 4 style
                Reference:
                    https://huggingface.co/transformers/usage.html

            QuestionAnsweringModelOutput
                Notes:
                    Output of the model_bart when answering a question
                Reference:
                    https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.QuestionAnsweringModelOutput

        :param str_question:
        :param str_context:
        :return:
        """

        # A dict containing a pyTorch Tensor of the question and tuple_given
        inputs = self.tokenizer.encode_plus(str_question,
                                            str_context,
                                            # Encode the sequences with the special tokens
                                            add_special_tokens=True,
                                            # Return pyTorch tensors
                                            return_tensors="pt",
                                            # Truncate tokens (make them smaller/delete)
                                            truncation=True
                                            )

        # pyTorch Tensor to list of ids where the ids represent the tokens
        input_ids = inputs["input_ids"].tolist()[0]

        # Tokenized representation of the id form of the question and tuple_given
        list_tokens_all = self.tokenizer.convert_ids_to_tokens(input_ids)

        # QuestionAnsweringModelOutput object
        question_answering_model_output: QuestionAnsweringModelOutput = self.model_question_answer(**inputs,
                                                                                                   return_dict=True)

        # Tensor score of the inputs tensor for the start of the answer
        # print("answer_start_scores")
        # print(type(question_answering_model_output.start_logits))
        # print(question_answering_model_output.start_logits)
        # print()

        # Tensor score of the inputs tensor for the end of the answer
        # print("answer_end_scores")
        # print(type(question_answering_model_output.end_logits))
        # print(question_answering_model_output.end_logits)
        # print()

        """
        Get the most likely beginning of answer with the argmax of the score
        (Get's the index of the start of the answer)
        """
        answer_start = torch.argmax(
            question_answering_model_output.start_logits
        )

        """
        Get the most likely end of answer with the argmax of the score
        (Get's the index of the end of the answer)
        """
        answer_end = torch.argmax(
            question_answering_model_output.end_logits) + 1

        # List of the ids based on the argmax index of the start of answer_start to answer_end
        list_ids_answer = input_ids[answer_start:answer_end]

        # List of tokens based on list_ids_answer
        list_tokens_answer = self.tokenizer.convert_ids_to_tokens(list_ids_answer)

        # String of the list_tokens_answer
        str_answer = self.tokenizer.convert_tokens_to_string(list_tokens_answer)

        return str_answer, list_tokens_answer, list_ids_answer, list_tokens_all, question_answering_model_output

    def print_answer_question(self, str_question: str, str_context: str, amount_space=0) -> Tuple[str,
                                                                                                  List[str],
                                                                                                  List[int],
                                                                                                  List[str],
                                                                                                  QuestionAnsweringModelOutput]:
        """
        Simple printer

        :param str_question:
        :param str_context:
        :param amount_space:
        :return:
        """
        str_answer, list_tokens_answer, list_ids_answer, list_tokens_all, question_answering_model_output = self.answer_question(
            str_question, str_context)

        print("{0:}Question: {1:}\n"
              "{0:}Answer: {2:}".format(" " * amount_space,
                                        str_question,
                                        str_answer))

        return str_answer, list_tokens_answer, list_ids_answer, list_tokens_all, question_answering_model_output


def example_documentation():
    text = r"""
    🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    TensorFlow 2.0 and PyTorch.
    """

    questions = [
        "How many pretrained models are available in Transformers?",
        "What does Transformers provide?",
        "Transformers provides interoperability between which frameworks?",
    ]

    model = QuestionAndAnsweringSystemHandler(MODEL_PRETRAINED_ALBERT_XLARGE_V2_SQUAD_V2)

    for question in questions:
        model.print_answer_question(question, text)


def example_pres():
    context = """On April 30, 1789, George Washington, standing on the balcony of Federal Hall on Wall Street in 
    New York, took his oath of office as the first President of the United States. “As the first of every thing, 
    in our situation will serve to establish a Precedent,” he wrote James Madison, “it is devoutly wished on my part, 
    that these precedents may be fixed on true principles.”"""

    print("Context len:", len(context))

    model = QuestionAndAnsweringSystemHandler(MODEL_PRETRAINED_ALBERT_XLARGE_V2_SQUAD_V2)

    question = "Who is the first president?"

    model.print_answer_question(question, context)


if __name__ == '__main__':
    # example_documentation()
    example_pres()
