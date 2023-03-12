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

"""

import wikipedia


def wiki_search():

    x = wikipedia.search("Obama")

    print(type(x))

    print(x)



if __name__ == '__main__':
    wiki_search()