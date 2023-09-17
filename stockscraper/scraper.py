# import requests
# from bs4 import BeautifulSoup

# base_url = "https://finance.yahoo.com/quote/"
# ticker = input()
# stock_url = ticker + "/news?p=" + ticker

# page =  requests.get(base_url)
# print(base_url + stock_url)

# soup = BeautifulSoup(page.content, "html.parser")

# print(soup)

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome('./chromedriver')

driver.get("https://www.finance.yahoo.com")

print(driver.title)