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

from selenium import webdriver
from selenium.webdriver import DesiredCapabilities
from selenium.webdriver.chrome.options import Options

PATH_CHROME_PROFILE_FOLDER = r'C:\Users\Joseph\AppData\Local\Google\Chrome\User Data'
OPTION_CHROME_PROFILE = "user-data-dir={}".format(PATH_CHROME_PROFILE_FOLDER)
# OPTION_CHROME_PROFILE_SPECIFIC = '--profile-directory=Profile 1'


class Handler:

    chrome_options = Options()

    def __init__(self):

        self.path_chrome_driver = r'.\selenium_handler\chromedriver.exe'
        self.path_chrome_driver = r'chromedriver.exe'

        self.load()

    def load(self):
        self.desired_caps = DesiredCapabilities().CHROME

        # Allow loading of pages
        # self.desired_caps["pageLoadStrategy"] = "normal"  # Wait for page to load always
        self.desired_caps["pageLoadStrategy"] = "none"  # Ignore page load and just load everything regardless

        # Create Selenium Driver Options
        self.chrome_options = Options()

        # Profile Folder (Folder where the profiles are)
        # Default killerjoseph245 is used if no specific profile is called
        self.chrome_options.add_argument(OPTION_CHROME_PROFILE)

        # Use specific profile
        # self.chrome_options.add_argument(OPTION_CHROME_PROFILE_SPECIFIC)

        # Headless chrome
        # self.chrome_options.add_argument("--headless")

        # uhh
        # self.chrome_options.add_argument("--disable-gpu")

        # Resolution
        # self.chrome_options.add_argument("--window-size=1920x1080")

        # Disable info bars (So it won't say that the browser is being controlled by Selenium) (DOES NOT WORK ANYMORE DUE TO SECURITY)
        self.chrome_options.add_argument("--disable-infobars")

        # Start Maximized
        self.chrome_options.add_argument("--start-maximized")

        # Full Screen FORCED, NO GOING BACK
        # self.chrome_options.add_argument("--kiosk")

        # Create Selenium Driver
        self.selenium_driver = webdriver.Chrome(
            desired_capabilities=self.desired_caps,
            executable_path=self.path_chrome_driver,
            chrome_options=self.chrome_options)

    def search(self, query):
        url = "http://www.google.com/search?q=" + query

        print(self.selenium_driver.get(url))


if __name__ == '__main__':
    x = Handler()

    x.search("Who are you?")
