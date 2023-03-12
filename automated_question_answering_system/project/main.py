"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 4/8/2021

Purpose:

Details:

Description:

Notes:

    Do not use
        pip install google-search-results


IMPORTANT NOTES:
    THE ORDER OF THE ARGUMENTS FOR THE SUMMARIZER MATTERS!

Explanation:

Reference:

"""

import colorama

from project.question_answer_system import (MODEL_PRETRAINED_ALBERT_XLARGE_V2_SQUAD_V2,
                                            MODEL_PRETRAINED_BERT_LARGE_UNCASED_WHOLE_WORD_MASKING_FINETUNED_SQUAD,
                                            QuestionAndAnsweringSystemHandler)
from project.scraper import get_context
from project.summarizer import Summarizer, MODEL_DISTILBART_CNN_12_6

THREADS_MAX = 5

AMOUNT_SEARCHES = 5
AMOUNT_SPACING = 5


class QuestionAndAnsweringSystem:

    def __init__(self,
                 dict_k_name_model_v_model_qash,
                 dict_k_name_model_d_model_summarizer):

        # Dict {Model name: Model}
        self.dict_k_name_model_v_model_qas = dict_k_name_model_v_model_qash

        # Dict {Model name: Model}
        self.dict_k_name_model_d_model_summarizer = dict_k_name_model_d_model_summarizer

        ###

        self.dict_k_name_model_v_model_pre_trained_qas = None

        self.dict_k_name_model_v_model_pre_trained_summarizer = None

    def load_models(self):
        self.dict_k_name_model_v_model_pre_trained_qas = load_dict_k_name_model_v_model(
            self.dict_k_name_model_v_model_qas,
            QuestionAndAnsweringSystemHandler)

        self.dict_k_name_model_v_model_pre_trained_summarizer = load_dict_k_name_model_v_model(
            self.dict_k_name_model_d_model_summarizer,
            Summarizer)

    def run(self):
        self.load_models()

        self.ask_question()

    def ask_question(self):
        while True:
            question = input("Enter Question: ")
            print("Question: {}".format(colorama.Fore.RED + question + colorama.Style.RESET_ALL))

            if question == "no":
                return

            context_aggregate, list_tuple_url_context = get_context(question, AMOUNT_SEARCHES)

            # TODO: FIX TO SUPPORT THREADING/MULTIPROCESSING
            # with ProcessPoolExecutor(max_workers=THREADS_MAX) as executor:
            #     # Future of the callable
            #     future_callables = [executor.submit(question_answer_system_albert.print_answer_question,
            #                                         question,
            #                                         tuple_given) for tuple_given in list_tuple_url_context]
            #
            #     # Add all the results of each process together
            #     for i in concurrent.futures.as_completed(future_callables):
            #         print(i)

            """
            Encoders with a Sequence2Sequence summarizer

            Notes:
                This is suboptimal...

            """
            # Loop print asking QuestionAnsweringSystem a question and getting an answer
            for tuple_pair in list_tuple_url_context:

                for name_model_qas, model_pre_trained_qas in self.dict_k_name_model_v_model_pre_trained_qas.items():

                    print("{}{}".format(" " * AMOUNT_SPACING, name_model_qas))

                    # t1 = threading.Thread(target=print_model_pre_trained_qas_tuple,
                    #                       args=[model_pre_trained_qas,
                    #                             question,
                    #                             tuple_pair],
                    #                       kwargs={"amount_space": AMOUNT_SPACING * 2})

                    dict_name_arg = {"Url": tuple_pair[0]}

                    result = print_model_pre_trained_qash(model_pre_trained_qas,
                                                          question,
                                                          tuple_pair[1],
                                                          dict_additional_print=dict_name_arg,
                                                          amount_space=AMOUNT_SPACING * 2)

                    for name_model_summarizer, model_pre_trained_summarizer in self.dict_k_name_model_v_model_pre_trained_summarizer.items():
                        print("{}{}".format(" " * AMOUNT_SPACING * 2, name_model_summarizer))

                        list_temp = [question, result[0]]

                        print_model_pre_trained_summarizer(model_pre_trained_summarizer,
                                                           list_temp,
                                                           amount_space=AMOUNT_SPACING * 3)
                    print()

            # AGGREGATE
            # for name_model_qas, model_pre_trained_qas in self.dict_k_name_model_v_model_pre_trained_qas.items():
            #
            #     print("{}{} on aggregate tuple_given".format(" " * AMOUNT_SPACING, name_model_qas))
            #
            #     for name_model_summarizer, model_pre_trained_summarizer in self.dict_k_name_model_v_model_pre_trained_summarizer.items():
            #         result = print_model_pre_trained_qash(model_pre_trained_qas,
            #                                               question,
            #                                               context_aggregate,
            #                                               amount_space=AMOUNT_SPACING * 2)
            #
            #         print("{}{}".format(" " * AMOUNT_SPACING * 2, name_model_summarizer))
            #
            #         list_temp = [question, result[0]]
            #
            #         print_model_pre_trained_summarizer(model_pre_trained_summarizer,
            #                                            list_temp,
            #                                            amount_space=AMOUNT_SPACING * 3)
            #         print()


def print_model_pre_trained_summarizer(summarizer: Summarizer, list_given, amount_space=0):
    str_result = (colorama.Fore.LIGHTBLUE_EX +
                  summarizer.summarize_list(
                      list_given,
                      question_and_answer=True) +
                  colorama.Style.RESET_ALL)

    print("{}Summarized answer: {}".format(" " * amount_space, str_result))


def print_model_pre_trained_qash(model: QuestionAndAnsweringSystemHandler,
                                 question: str,
                                 context: str,
                                 dict_additional_print: dict = None,
                                 amount_space: int = 10):
    result = model.answer_question(
        question,
        context)

    str_answer = result[0]
    list_tokens_answer = result[1]
    list_ids_answer = result[2]
    list_tokens_all = result[3]
    question_answering_model_output = result[4]

    if dict_additional_print:
        for key, value in dict_additional_print.items():
            print("{0:}{1:}: {2:}".format(" " * amount_space, key, value))

    print("{0:}Answer: {1:}".format(
        " " * amount_space,
        (colorama.Fore.BLUE +
         str_answer +
         colorama.Style.RESET_ALL))
    )

    return str_answer, list_tokens_answer, list_ids_answer, list_tokens_all, question_answering_model_output


def load_dict_k_name_model_v_model(dict_k_name_model_v_model: dict, loader):
    dict_k_name_model_v_model_pre_trained = {}

    for key, value in dict_k_name_model_v_model.items():
        model_pre_trained = loader(value)

        dict_k_name_model_v_model_pre_trained[key] = model_pre_trained

        print("{} loaded\n".format(key))

    return dict_k_name_model_v_model_pre_trained


def main():
    """
    Bart was not meant for QA!!
    """

    dict_k_name_model_v_model_qash = {
        f"Albert {MODEL_PRETRAINED_ALBERT_XLARGE_V2_SQUAD_V2}": MODEL_PRETRAINED_ALBERT_XLARGE_V2_SQUAD_V2,
        f"Bert {MODEL_PRETRAINED_BERT_LARGE_UNCASED_WHOLE_WORD_MASKING_FINETUNED_SQUAD}": MODEL_PRETRAINED_BERT_LARGE_UNCASED_WHOLE_WORD_MASKING_FINETUNED_SQUAD,
        # f"Bart {MODEL_DISTILBART_CNN_12_6}": MODEL_DISTILBART_CNN_12_6,
    }

    dict_k_name_model_d_model_summarizer = {
        f"Bart {MODEL_DISTILBART_CNN_12_6}": MODEL_DISTILBART_CNN_12_6
    }

    question_and_answering_system = QuestionAndAnsweringSystem(
        dict_k_name_model_v_model_qash,
        dict_k_name_model_d_model_summarizer)

    question_and_answering_system.run()


if __name__ == '__main__':
    main()
