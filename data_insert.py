from sklearn.datasets import fetch_20newsgroups
import requests
import time
import json

twenty_train = fetch_20newsgroups()

def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

url = "http://localhost:8000/elmo"
data = twenty_train["data"][12:-1]
header = {"Content-Type":"application/json"}

for i in list(divide_chunks(data, 2)):
    response = requests.request("POST", url, data=json.dumps({"text": i}), headers=header)
    print(response.json())
    time.sleep(10)