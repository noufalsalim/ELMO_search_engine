from flask import Flask, jsonify, request
import requests
import pymongo
from sklearn.metrics.pairwise import cosine_similarity as cs
import numpy as np
from operator import itemgetter
import json

client = pymongo.MongoClient("localhost", 27017)
db = client.symanticsearch

app = Flask(__name__)

def emb(data):
    serving_payload = {"inputs":{"text": data}}
    serving_header = {"Content-Type":"application/json"}
    serving_url = "http://localhost:9000/v1/models/elmo:predict"

    response = requests.request("POST", serving_url, headers=serving_header, json=serving_payload)
    
    try:
        result = np.mean(response.json()["outputs"]["elmo"], axis=1).tolist()
        word_emb = np.mean(response.json()["outputs"]["word_emb"], axis=1).tolist()
        seq_len = [response.json()["outputs"].get("sequence_len", None)][0]
    except KeyError:
        result = None
        word_emb = None
        seq_len = None
    return result, seq_len, word_emb

@app.route("/", methods=['GET'])
def status():
    return jsonify(status="connected")

@app.route("/elmo", methods=["POST"])
def elmo():
    data = request.get_json()
    import time
    start = time.time()
    result, seq_len, word_emb = emb(data["text"])

    if result:
        print("Embedding calculated in {} seconds".format(str(time.time()-start)))
        for i, j, seq, word_em in zip(data["text"], result, seq_len, word_emb):
            dump = {}
            dump["text"] = i
            dump["result"] = j
            dump["sequence_len"] = seq
            dump["word_emb"] = word_em
            db.embedding.insert_one(dump)
            print("data inserted")
    else:
        print("skipped calculating embedding")
        result = []

    return jsonify(vector=result)

@app.route("/search", methods=["GET"])
def symsearch():
    query = request.args.get("query")
    result = emb([query])
    possible = []
    trained_data = db.embedding.find()
    for value in trained_data:
        out = cs(result, [value["result"]])
        if out[0][0] >= 0.2:
            similar = {"text": value["text"], "similarity": out[0][0]}
            possible.append(similar)
    searchout = sorted(possible, key=itemgetter('similarity'), reverse=True) 

    return jsonify(result = searchout)

if __name__ == "__main__":
    print("""
    run the app with gunicorn server
    """)