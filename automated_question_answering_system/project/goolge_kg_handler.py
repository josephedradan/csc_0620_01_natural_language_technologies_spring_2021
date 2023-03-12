"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 5/20/2021

Purpose:

Details:

Description:

Notes:

IMPORTANT NOTES:

Explanation:

Reference:
    https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da

"""
import json

import en_core_web_lg
import requests

nlp = en_core_web_lg.load()
print("Spacy Loaded...")
print()


class KGHanlder:
    def __init__(self):
        pass


def _kg_search(query):
    """
    Use Google Knowledge graph API to search for query

    **** query should be the most basic representation of your sentence

    :param query:
    :return:
    """
    with open("GOOGLE_KG_API_KEY.txt") as f:
        api_key = f.read()

    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'query': query,
        'limit': 3,
        'indent': False,
        'key': api_key,
    }
    # url = service_url + '?' + urlencode(params)
    # print(url)
    # dict_result = json.loads(urlopen(url).read())

    response = requests.get(service_url,
                            params)

    text_response = response.content

    # print("\n")
    # print(text_response)
    # print("\n")

    dict_result = json.loads(text_response)

    # print("\n")
    # print(dict_result)
    # print("\n")
    # pprint(dict_result)

    return dict_result


def search_kg_query_for_urls(query):
    """
    Get a list of the URLS

    :param query:
    :return:
    """
    # IF EMPTY STRING
    if query == "":
        return []

    response = _kg_search(query)
    list_url = []

    # print("-" * 100)
    # print(response)
    # print("-" * 100)

    element: dict
    for element in response['itemListElement']:
        # print(element)
        # print(element['result']['name'] + ' (' + str(element['resultScore']) + ')')


        try:
            # url = element.get("result").get("detailedDescription").get("url")
            url = element["result"]["detailedDescription"]["url"]
            list_url.append(url)

            # print("Result")
            # print(element)

            temp = str(element['result']['name'] + ' (' + str("{:.4f}".format(element['resultScore'])) + ')')

            print("\t{:<50}{}".format(str(temp), url))

            # print("End")
        except Exception as e:
            pass

        # result = element.get("result")
        # if result is not None:
        #     detailed_description = result.get("detailedDescription")
        #     if detailed_description is not None:
        #         print("RESULT2")
        #
        #         url = result.get("url")
        #         if url is not None:
        #             print("RESULT4")
        #
        #             list_url.append(url)

    return list_url


def search_kg_query_for_urls_spacey(query):
    """
    Use spacey to get the entities and the nouns into a string

    :param query:
    :return:
    """
    document = nlp(query)

    list_str_noun_chunk = [str(i) for i in document.noun_chunks]
    # print("list_str_noun_chunk:", list_str_noun_chunk)

    entities = document.ents
    list_str_entity = [str(i) for i in entities]
    # print("list_str_entity:", list_str_entity)

    str_entities = " ".join(list_str_entity)
    # print("str_entities:", str_entities)

    list_temp = []
    list_temp.extend(list_str_entity)
    list_temp.extend(list_str_noun_chunk)
    # print("list_temp", list_temp)

    list_together = list(set(list_temp))

    str_together = " ".join(list_together)

    print(f"Google Knowledge Graph Results for \"{str_together}\":")

    return search_kg_query_for_urls(str_together)


if __name__ == '__main__':
    # x = search_kg_query_for_urls("Who is the first president of the US?")
    # x = search_kg_query_for_urls_spacey("Who is the first president of the US?")
    # x = search_kg_query_for_urls_spacey("Who is Jeff Bezos")
    x = search_kg_query_for_urls_spacey("When did Perseverance land on mars?")

    for i in x:
        print(i)
