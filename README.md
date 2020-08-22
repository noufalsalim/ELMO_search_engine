## ELMO based search engine with mongoDB and Flask served with tensorflow

This is an elmo based search engine where you can vectorize your text using elmo embedding and save the values in MongoDB to search any particular text later with symantic similarity 

### technologies used
- ELMO
- tensorflow serving
- flask
- MongoDB

### installation and running tensorflow server

    apt install curl

    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

    apt-get update && apt-get install tensorflow-model-server

    apt-get upgrade tensorflow-model-server

    python model_prep.py

    saved_model_cli show --dir $(pwd)/model/1 --all

    tensorflow_model_server --model_base_path=$(pwd)/model --rest_api_port=9000 --model_name=elmo


#### TODO
Dockerfile adding