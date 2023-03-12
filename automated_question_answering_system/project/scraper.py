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

    Scrapy
        Notes:
            I'm not going to write a scrapper from the ground up...

        Reference:
            https://github.com/scrapy/scrapy
"""

import re
# pip install google
from pprint import pprint
from typing import Tuple, List

import requests
from bs4 import BeautifulSoup

from project.googlesearch.googlesearch import search
from project.goolge_kg_handler import search_kg_query_for_urls_spacey


def get_google_search_urls(query: str, amount):
    list_searches = search(query, tld="com", num=amount, stop=amount, pause=2)
    return list_searches


def get_google_search_urls_v2(query: str, amount):
    """
    https://stackoverflow.com/questions/25471450/python-getting-all-links-from-a-google-search-result-page

    :param query:
    :param amount:
    :return:
    """

    search = "{}{}".format("https://www.google.dz/search?q=", query)

    page = requests.get(search)

    page_actual = page.content

    bs4_object = BeautifulSoup(page_actual, "html.parser", from_encoding="iso-8859-1")

    list_link_all = bs4_object.find_all("a")

    # for link in bs4_object.find_all("a", href=re.compile("(?<=/url\?q=)(htt.*://.*)")):
    #     print(re.split(":(?=http)", link["href"].replace("/url?q=", "")))

    # print(bs4_object.prettify())

    for i in list_link_all:
        print(i)


def scrape_page_basic(url: str) -> str:
    """
    Timeout
        https://stackoverflow.com/questions/45267003/python-requests-hanging-freezing

    :param url:
    :return:
    """
    # https://realpython.com/beautiful-soup-web-scraper-python/
    try:
        page = requests.get(url, timeout=3)
    except Exception as e:
        # traceback.print_exc()
        # print(e)
        print("Requests timeout reached for {}".format(url))

        return ""

    # print(page.content)
    # Solve the Some characters could not be decoded, and were replaced with REPLACEMENT CHARACTER. issues use from_encoding
    bs4_object = BeautifulSoup(page.content, "html.parser", from_encoding="iso-8859-1")
    # bs4_object = BeautifulSoup(page.text, "lxml")

    # Get only paragraphs because they contain proper sentences
    paragraphs = bs4_object.find_all("p", recursive=True)  # Do not do text=True, it will miss out

    # print(bs4_object)

    str_paragraph_all = ""

    for paragraph_tag_full in paragraphs:

        str_word = paragraph_tag_full.text
        # print(str_word)

        # str_word = re.sub(r"\s", " ", str_word)
        # str_word = re.sub(r"( +)", " ", str_word)
        # str_word = str_word.split()

        # print("sdfs:",len(re.findall("[\r\n\t\f\v]{3,}", str_word)))
        # print(str_word)

        # Ignore these types of white space [\r\n\t\f\v] with 3 or more of them
        if len(re.findall("[\r\n\t\f\v]{3,}", str_word)) > 0:
            continue

        count_word = len(str_word.split(" "))

        # print(count_word)

        if count_word < 4:
            continue

        str_paragraph_all += " " + str_word

    str_paragraph_all = str_paragraph_all.strip()
    # print(str_paragraph_all)

    return str_paragraph_all


def get_context(query, amount=1) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Given a question, will search via google.


    :param query:
    :param amount:
    :return:
    """
    query = "\"" + query + "\""

    list_searches: list = list(get_google_search_urls(query, amount))

    print("Google Search:")
    for i in list_searches:
        print("\t{}".format(i))
    print()

    list_searches_kg: list = list(search_kg_query_for_urls_spacey(query))
    print()

    # for i in list_searches_kg:
    #     print("\t", i)
    # print()

    list_searches.extend(list_searches_kg)

    list_tuple_url_context = []

    str_context_full = ""

    for url in list_searches:
        # print("URL:", url)

        str_paragraph_all = scrape_page_basic(url)

        list_tuple_url_context.append((url, str_paragraph_all))

        str_context_full += " " + str_paragraph_all

        # print()

    # Strip spaces
    str_context_full = str_context_full.strip()

    # replace multiple spaces
    str_context_full = re.sub(r"\s", " ", str_context_full)
    str_context_full = re.sub(r"( +)", " ", str_context_full)
    str_context_full = str_context_full.strip()

    return str_context_full, list_tuple_url_context


def example_google():
    str_context_full, list_tuple_url_context = get_context("Who founded Google?", 5)
    print()
    # print(text)
    print()

    for i in list_tuple_url_context:
        pprint(i)
        print()


def example_president():
    str_context_full, list_tuple_url_context = get_context("Who was the first president", 5)
    print()
    print(str_context_full)
    print()

    for i in list_tuple_url_context:
        pprint(i)
        print()


if __name__ == '__main__':
    # example_google()
    # example_president()

    pass
    # get_google_search_urls_v2("Who was the first president", 2)
