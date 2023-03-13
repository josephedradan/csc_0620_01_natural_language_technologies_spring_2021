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
    Google API
        https://developers.google.com/webmaster-tools/search-console-api-original/v3/quickstart/quickstart-python


        KG API KEY
        AIzaSyDBG110eS9kiHuj1DCEUxHlpgn_w_ol9Ro  # DEAD

        ID
        972097831436-1g557tlmhe2clst251mqn1ing8mokefj.apps.googleusercontent.com  # DEAD

        SECRET
        Dtk9wWLNRyyZzktDb30BhQvX  # DEAD
"""

# !/usr/bin/python

import httplib2

from apiclient.discovery import build
from oauth2client.client import OAuth2WebServerFlow

# Copy your credentials from the console
CLIENT_ID = '972097831436-1g557tlmhe2clst251mqn1ing8mokefj.apps.googleusercontent.com'  # DEAD
CLIENT_SECRET = 'Dtk9wWLNRyyZzktDb30BhQvX'  # DEAD

# Check https://developers.google.com/webmaster-tools/search-console-api-original/v3/ for all available scopes
OAUTH_SCOPE = 'https://www.googleapis.com/auth/webmasters.readonly'

# Redirect URI for installed apps
REDIRECT_URI = 'urn:ietf:wg:oauth:2.0:oob'

# Run through the OAuth flow and retrieve credentials
flow = OAuth2WebServerFlow(CLIENT_ID, CLIENT_SECRET, OAUTH_SCOPE, REDIRECT_URI)
authorize_url = flow.step1_get_authorize_url()
print('Go to the following link in your browser: ' + authorize_url)
code = raw_input('Enter verification code: ').strip()
credentials = flow.step2_exchange(code)

# Create an httplib2.Http object and authorize it with our credentials
http = httplib2.Http()
http = credentials.authorize(http)

webmasters_service = build('searchconsole', 'v1', http=http)

# Retrieve list of properties in account
site_list = webmasters_service.sites().list().execute()

# Filter for verified websites
verified_sites_urls = [s['siteUrl'] for s in site_list['siteEntry']
                       if s['permissionLevel'] != 'siteUnverifiedUser'
                       and s['siteUrl'][:4] == 'http']

# Print the URLs of all websites you are verified for.
for site_url in verified_sites_urls:
    print(site_url)
    # Retrieve list of sitemaps submitted
    sitemaps = webmasters_service.sitemaps().list(siteUrl=site_url).execute()
    if 'sitemap' in sitemaps:
        sitemap_urls = [s['path'] for s in sitemaps['sitemap']]
        print("  " + "\n  ".join(sitemap_urls))
