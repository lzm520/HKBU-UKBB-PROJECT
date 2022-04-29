import csv
import os
import re
from lxml import etree
import numpy as np
import requests
from selenium import webdriver
from collections import defaultdict


# 从ukb4190.html文件中将各个字段的字段类型抽取出来(此程序在服务器端不能读取ukb41910.html文件，但在本地电脑跑可以)
def Cols_type_extraction():
    fp = open('../../ukb41910.html', 'r')
    outfile = open('../../data/cols_type.txt', 'w')
    f = fp.read()
    fp.close()

    html = etree.HTML(f)
    contents_rows = html.xpath('/html/body/table[2]/tr')[2:]
    uids = []
    for i, row in enumerate(contents_rows):
        if np.mod(i, 5000) == 0:
            print('iterated entries:', i)

        row_content = etree.tostring(row, encoding='utf-8').decode('utf-8')
        uid = re.search(r'<a.*?>(.*)</a>', row_content).group(1).split('-')[0]
        if uid in uids:
            continue
        uids.append(uid)
        type = re.search(r'<span.*?>(\w*).*?</span>', row_content).group(1)
        type.strip()
        uid.strip()
        outfile.write(type + '\t' + uid)
        outfile.write('\n')
    outfile.close()


# 从ukbb官网中爬取字段的类型（由于此爬虫使用selenium，需要配置环境，因此最好在本地爬取）
def Extract_data_category_from_ukbWebsite():
    path = '../../data/data_category/'
    if not os.path.exists(path):
        os.mkdir(path)
    url = 'https://biobank.ndph.ox.ac.uk/showcase/cats.cgi'
    driver = webdriver.Chrome()
    driver.get(url)
    urls = []
    big_category = []
    A = driver.find_elements_by_xpath('//div[contains(@class,"tabberlive")]/div[3]/table/tbody/tr')[1:]
    for c in A:
        d = c.find_elements_by_xpath('./td/a')[1]
        category = d.get_attribute('innerHTML').strip()
        u = d.get_attribute('href')
        urls.append(u)
        big_category.append(category)
    for category in big_category:
        if not os.path.exists(path + category):
            os.mkdir(path + category)
    for i, url in enumerate(urls):
        driver.get(url)
        trs = driver.find_elements_by_xpath(
            '//div[contains(@class, "tabbertab")]/table[contains(@summary, "List of data-fields")]/tbody/tr')[1:]
        with open(path + big_category[i] + '/' + big_category[i] + '.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            for tf in trs:
                a = tf.find_elements_by_xpath('./td/a')
                field_id = a[0].get_attribute('innerHTML')
                description = a[1].get_attribute('innerHTML')
                category = a[2].get_attribute('innerHTML')
                writer.writerow([field_id, description, category])
    driver.quit()


if __name__ == '__main__':
    Extract_data_category_from_ukbWebsite()
