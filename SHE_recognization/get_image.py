# -*- coding: utf-8 -*-
import os
import uuid
import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

# 下载图片
def get_picture(url, person, key_word):

    #利用PhantomJS加载网页
    browser = webdriver.PhantomJS()
    browser.set_page_load_timeout(30) # 最大等待时间为30s
    #当加载时间超过30秒后，自动停止加载该页面
    try:
        browser.get(url)
    except TimeoutException:
        browser.execute_script('window.stop()')
    source = browser.page_source #获取网页源代码
    browser.quit()
    #解析网页，获取下载图片的网址
    soup = BeautifulSoup(source,'lxml')
    images = soup.find_all('img')

    # 下载任务图片
    for image in images:
        if 'alt' in image.attrs.keys():
            if key_word in image['alt']:
                print(image)
                urllib.request.urlretrieve(image['src'], 'E://face_data/%s_raw_image/%s.jpg'%(person, uuid.uuid1()) )

def main():

    SHE_dict = {'Ella': '陈嘉桦', 'Hebe': '田馥甄', 'Selina': '任家萱'}
    directory = 'E://face_data'
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            person = file.split(r'.')[0]
            with open('%s/%s'%(directory,file), 'r') as f:
                urls = [string.strip() for string in f.readlines()]

            print('Getting pictures of %s'%person)
            # 利用并发下载电影图片
            executor = ThreadPoolExecutor(max_workers=10)  # 可以自己调整max_workers,即线程的个数
            # submit()的参数： 第一个为函数， 之后为该函数的传入参数，允许有多个
            future_tasks = [executor.submit(get_picture, url, person, SHE_dict[person]) for url in urls]
            # 等待所有的线程完成，才进入后续的执行
            wait(future_tasks, return_when=ALL_COMPLETED)

    print('图片下载完毕！')

main()