import requests
import time
import csv
import re

links=[]
id = 1
with open('cik_list.csv') as fp:
    fp.readline()
    rows = csv.reader(fp)
    for row in rows:
       links.append([str(row[1].split()[0]+str(id)),str(row[7])])
       id+=1

for link in links:
    url = link[1]
    r = requests.get(url)
    raw_text = r.text

    sec_1 = re.findall("\nITEM \d+\. MANAGEMENT'S DISCUSSION AND ANALYSIS .*?(?:\nITEM|\nPART)", raw_text, re.S)
    sec_2 = re.findall("\nITEM \d+\. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK .*?(?:\nITEM|\nPART)", raw_text, re.S)
    sec_3 = re.findall("\nITEM \d+\. RISK FACTORS .*?(?:\nITEM|\nPART)", raw_text, re.S)
    
    if len(sec_1):
        with open(link[0]+'.txt','a') as fp:
            fp.writelines(sec_1[0])
    if len(sec_2):
        with open(link[0]+'.txt','a') as fp:
            fp.writelines(sec_2[0])
    if len(sec_3):
        with open(link[0]+'.txt','a') as fp:
            fp.writelines(sec_3[0])
    
    time.sleep(4)
    
