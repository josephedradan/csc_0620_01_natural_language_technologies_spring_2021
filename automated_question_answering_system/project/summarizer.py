"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 4/8/2021

Purpose:

Details:

Description:

Notes:

IMPORTANT NOTES:

Explanation:

Reference:
    Huggingface Summarization
        Notes:
            Basic code
        Reference:
            https://huggingface.co/transformers/task_summary.html#summarization

    BartForConditionalGeneration
        Notes:
            Using bart
        Reference:
            https://huggingface.co/transformers/model_doc/bart.html#transformers.BartForConditionalGeneration

"""
import re

from joseph_library.decorators import timer
from transformers import (AutoTokenizer,
                          AutoModelForMaskedLM,
                          BartForConditionalGeneration,
                          PreTrainedTokenizerFast)

MODEL_DISTILBART_CNN_12_6 = "sshleifer/distilbart-cnn-12-6"


@timer
class Summarizer:

    def __init__(self, name_model_huggingface: str = MODEL_DISTILBART_CNN_12_6):

        # Model BartForConditionalGeneration
        self.model_bart: BartForConditionalGeneration = AutoModelForMaskedLM.from_pretrained(name_model_huggingface)

        # Tokenizer
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(name_model_huggingface)

    def summarize(self, text, question_and_answer=False) -> str:
        # print("TEXT", text)
        if question_and_answer:
            """
            Prevent output sequence to be bigger than the length of the given text.
            I have no proof that all summarization sizes are less than the given text.
            """
            max_length = len(text)

            """
            Assuming that the given text is small (which should be technically true because you have a question and an
            answer) then we can use more compute time to generate a better sequence
            
            """
            early_stopping = False

        else:
            max_length = 20
            early_stopping = True

        # pyTorch Tensor representation of input text in id form
        inputs = self.tokenizer.encode(text,
                                       # Return pyTorch tensors
                                       return_tensors="pt",
                                       # max_length=1024,
                                       # Truncate tokens (make them smaller/delete)
                                       truncation=True
                                       )

        """
        
        Notes:
            The number for num_beams results in the amount of words that will be used to
            formulate a sequence of words that depend on each other based on a probability.
            Basically you select num_beams of words then use probability by loop through all
            the other words to formulate a sequence that is a sentence 
            (num_beams * size of vocab = amount of iterations over your vocab, also big memory usage)
            
        Reference:
            C5W3L03 Beam Search
                Notes:
                    Andrew Ng
                Reference:
                    https://www.youtube.com/watch?v=RLWuzLLSIgw
                    
            C5W3L04 Refining Beam Search
                Notes:
                    Use normalize by dividing by the amount of words in vocab^alpha * Summation of log of prob
                    where alpha is a hyper parameter around 0.7 to start but you need to tune it (it's a hack not
                    a heuristic, there is no math to back up 0.7, it was just nice to use ).
                    
                    beam width B big -> better result, slow, more ram
                    beam width B small -> worst, faster
                    
                    Production: 10
                    Commercial: 100 to 3000
                    
                    "Beam search runs faster but is not guaranteed to find exact max for arg max P(y|x)"
                    
                Reference:
                    https://www.youtube.com/watch?v=gb__z7LlN_4
                    
            C5W3L05 Error Analysis of Beam Search
                Notes:
                    Basically for beam,
                        Beam choose algo result prob when human result prob is better then Beam is at fault
                        RNN choose algo result when human result is better then RNN is at fault 
                Reference:
                    https://www.youtube.com/watch?v=ZGUZwk7xIwk
            
            generate
                Notes:
                    generate method list_given and kwargs
                    
                    top_k 
                        The number of highest probability vocabulary tokens to keep for top-k-filtering.
                        It's the top_k amount of tokens with the highest probabilities. 
    
                Reference:
                    https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate
            
        """
        # pyTorch Tensor representation of output text in id form
        outputs = self.model_bart.generate(inputs,  # Ignore type expectation, this should be a pyTorch Tensor
                                           # Max length of sequence, default == 20
                                           max_length=max_length,
                                           # Min length of sequence, default == 10
                                           min_length=0,
                                           # >1 == encourage bigger sequence, <1 == encourage smaller sequence
                                           length_penalty=1.0,  # was 2
                                           # Number of beams for beam search (Look at notes for more detail).
                                           num_beams=5,
                                           # Stop beam search when num_beams (amount) sentences are done
                                           early_stopping=early_stopping  # False would imply a better summarization
                                           )

        """
        List containing the actual text
        
        Notes:
            Take the ids from the "outputs" tensor and tokenized the ids to text 
        
        """
        summarization = [self.tokenizer.decode(id,
                                               skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True
                                               ) for id in outputs
                         ]

        # Return the summarized text
        return summarization[0]

    def summarize_list(self, list_given, question_and_answer=False):
        """

        FIXME: VERY NOT CLEAN CODE

        :param list_given:
        :param question_and_answer:
        :return:
        """

        # If using question_and_answer mode
        if question_and_answer:
            str_temp = " ".join(list_given)

            # print(len(tokens))
            # print(tokens)
            # print(list_given)

            # Empty string arg
            for i in list_given:
                if i == "":
                    # warnings.warn("Args contained \"\"")
                    return ""

            # If invalid information string
            if len(re.findall(r"\[CLS]|\[SEP]", str_temp)) > 0:
                # warnings.warn("Args contained a non answer token")
                return ""

            # If only 1 arg
            if len(list_given) < 2:
                # warnings.warn("Not enough list_given to form a correct sentence")
                return ""

        return self.summarize(" ".join(list_given), question_and_answer=question_and_answer)


def example_summarize_q_a():
    question = "Who founded Google?"
    answer = "larry page and sergey brin"

    both = " ".join([question, answer])

    test = Summarizer()

    result = test.summarize(both)

    print(result)


def example_summarize_args_qa():
    print(f"Testing {example_summarize_args_qa.__name__}")
    test = Summarizer()

    result = test.summarize_list(["Are functions first class in java?", "[CLS]"], question_and_answer=True)
    print(result)


def example_random_text():
    print(f"Testing {example_random_text.__name__}")
    test = Summarizer()

    x = """Realistically, most of the time you could just go through a website manually and grab the data ‘by hand’
    using copy and paste, but in a lot of cases that would take you many hours of manual work, which could end up
    costing you a lot more than the data is worth, especially if you’ve hired someone to do the task for you. Why
    hire someone to work at 1–2 minutes per query when you can get a program to perform a query automatically every
    few seconds? For example, let’s say that you wish to compile a list of the Oscar winners for best picture, along
    with their director, starring actors, release date, and run time. Using Google, you can see there are several sites
    that will list these movies by name, and maybe some additional information, but generally you’ll have to follow
    through with links to capture all the information you want.
    """

    print(test.summarize(x))


if __name__ == '__main__':
    example_summarize_args_qa()
    # example_summarize_q_a()
