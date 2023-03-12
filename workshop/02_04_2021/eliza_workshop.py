"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 2/8/2021

Purpose:
    Do the workshop due on Feb 4th

Details:

Description:
    Simple wrapper over nltk.chat.eliza module

Notes:
    Very very very basic Chat bot.

IMPORTANT NOTES:

Explanation:

Reference:
    Source code for nltk.chat.eliza
        Notes:
            Use this a base

        Reference:
            https://www.nltk.org/_modules/nltk/chat/eliza.html

"""
from collections import defaultdict
from typing import Union, Sequence, Tuple

from nltk.chat import eliza
from nltk.chat.util import Chat, reflections


class ElizaV2:
    """
    A Simple chat bot wrapper over the eliza in the nltk library

    """

    def __init__(self, pairs_given: Sequence, reflections_given: dict) -> None:
        """
        Makes a modified version of Eliza from the NLTK Library as an object

        :param pairs_given: pairs (E.g. "Go die" -> "I will not die!")
        :param reflections_given: reflections (E.g. "I am" -> "You are")
        """

        # The chat bot
        self.eliza_chatbot: Union[None, Chat] = None

        # Dict key user_response and list that should contain strings known as the responses
        self.dict_user_response_responses = defaultdict(list)

        # The new pairs list based on the given paris and additional user pairs
        self.pairs = []

        # The reflections for some combinations of words
        self.reflections = reflections_given

        # Subroutine to convert pairs a dict to allow for additional entries to existing pairs
        self._pairs_to_dict_user_response_responses(pairs_given)

    def _pairs_to_dict_user_response_responses(self,
                                               pairs_given: Sequence[Sequence[Union[str, Sequence[str]]]]) -> None:
        """
        Converts the pairs_given to a dict to allow for additional inputs responses for any given user_response

        :param pairs_given: A Sequence (most likely a tuple) containing a Sequence (most likely a tuple) with the first
                            object being a string and the second being a Sequence (most likely a tuple) with strings
        :return: None
        """

        # Loop through the pairs sequence
        for pair in pairs_given:
            # Add the user_response and its corresponding responses to the dict via list.extend
            self.dict_user_response_responses[pair[0]].extend(pair[1])

    def add_pair(self, user_response: str, responses: Union[list, str]) -> None:
        """
        Adds new user_response and responses pairs or just responses to the dict.
        response can be a list of response or just a single response like a string

        :param user_response: What the user says to the bot
        :param responses: The bot's responses
        :return: None
        """

        if isinstance(responses, list):
            self.dict_user_response_responses[user_response].extend(responses)

        elif isinstance(responses, str):
            self.dict_user_response_responses[user_response].append(responses)

    def _dict_user_response_responses_to_pairs(self):
        """
        Convert the dict user_response and it's corresponding responses to a list that is sorted with the
        longest user_response at the top. This method of sorting makes the most complex statements checked
        first implying that there are conflicting similarities among the user responses.

        :return: None
        """

        # Resen the pairs list
        self.pairs = []

        # This should be the last response that should catch all responses if all the use user_response strings failed
        user_response_catch_all: Union[Tuple[str, Sequence], None] = None

        # Loop through dict of user_response and responses
        for user_response, responses in self.dict_user_response_responses.items():

            # Check if user_response is the generic catch all user_response
            if user_response == r"(.*)":
                # Set user_response_catch_all
                user_response_catch_all = (user_response, responses)

                # Skip this iteration
                continue

            # Add the user_response, responses pair to the pairs list
            self.pairs.append((user_response, responses))

        # Default sort is by ascending order, use reverse and sort by length of user_response
        self.pairs.sort(reverse=True, key=lambda e: len(e[0]))

        # Add the last possible generic user_response
        self.pairs.append(user_response_catch_all)

        # Debug print
        # pprint(self.pairs, indent=5)

    def initialize(self) -> None:
        """
        Create the chat bot using the pairs and the reflections

        :return: None
        """

        # Convert the dict_user_response_responses to a list (pairs)
        self._dict_user_response_responses_to_pairs()

        # Create the chat bot object with the pairs and the reflections
        self.eliza_chatbot = Chat(self.pairs, self.reflections)

    def run(self) -> None:
        """
        Run the chat bot using the converse() method

        :return: None
        """
        print("Please start a conversation.")
        self.eliza_chatbot.converse()


if __name__ == '__main__':
    # Create the chat bot
    eliza = ElizaV2(eliza.pairs, reflections)

    # Add custom pairs
    eliza.add_pair(r"The finals week has been tough",
                   ["I hear you. Finals week is tough for many. Do you want to tell me more?",
                    "Sucks to suck!"])

    eliza.add_pair(r"Can you do my (.*)\?", ["I am incapable of doing %1!",
                                             "I can't do %1 right now, I am incapable of thinking."])

    eliza.add_pair(r"What do you think of (.*)\?", ["I'm not sure about %1, why don't you google it.",
                                                    "I don't know what %1 is."])

    eliza.add_pair(r"How can you be improved?", ["Integrate me with BERT!",
                                                 "Use a Deep Learning approach.",
                                                 "Why not use spacey...",
                                                 "Maybe use tensorflow..."])

    # Create the actual bot
    eliza.initialize()

    # Run the conversation
    eliza.run()
