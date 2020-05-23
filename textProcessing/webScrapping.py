import pandas as pd
import time
from selenium.webdriver import Chrome
from selenium import webdriver
import json
driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
browser = Chrome()
url = "https://www.instagram.com/"

def recent_posts(username):
    """With the input of an account page, scrape the 25 most recent posts urls"""
    url = "https://www.instagram.com/" + username + "/"
    browser = Chrome()
    browser.get(url)
    post = 'https://www.instagram.com/p/'
    post_links = []
    #xpath_comment = '//*[@id="react-root"]/section/main/div/header/section/ul/li[1]/a/span'
    #comment = browser.find_element_by_xpath(xpath_comment)
    #comment = comment.text
    #num_of_posts = int(comment)
    while len(post_links) < 150:
        links = [a.get_attribute('href') for a in browser.find_elements_by_tag_name('a')]
        for link in links:
            if post in link and link not in post_links:
                post_links.append(link)
        scroll_down = "window.scrollTo(0, document.body.scrollHeight);"
        browser.execute_script(scroll_down)
        time.sleep(10)
    else:
        return post_links

def insta_details(urls):
    """Take a post url and return post details"""
    browser = Chrome()
    post_details = []
    for link in urls:
        browser.get(link)
        xpath_comment = '//*[@id="react-root"]/section/main/div/div[1]/article/div[2]/div[1]/ul/div/li/div/div/div[2]/span'
        try:
            comment = browser.find_element_by_xpath(xpath_comment)
            comment = comment.text
            post_details.append(comment)
        except:
            continue

        time.sleep(10)
    return post_details


posts = insta_details(recent_posts('username'))
print(posts)
with open("username.json", "w") as write_file:
    json.dump(posts, write_file)