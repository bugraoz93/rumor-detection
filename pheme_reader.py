import os
import json
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import http.client
dataset_directory = "/home/cluster/PycharmProjects/data"
DS_STORE = '.DS_Store'
directories = os.listdir(dataset_directory)
directories.remove('README')

es = Elasticsearch(hosts="localhost:9200")
http.client._MAXHEADERS = 5000

# directories.remove(DS_STORE)


def read_data_to_dir():
    data = {}
    print("Reading Data Started!!")
    for directory in directories:
        event_name = directory
        print("Event: " + str(event_name))
        data[event_name] = {'rumor_tweets': [], 'non_rumor_tweets': []}
        event_dir_path = os.path.join(dataset_directory, directory)
        print("event path: " + str(event_dir_path))
        os.chdir(event_dir_path)
        clean_dirs = os.listdir(os.curdir)

        if clean_dirs.__contains__(DS_STORE):
            clean_dirs.remove(DS_STORE)

        rumor_dir = clean_dirs[0]
        non_rumor_dir = clean_dirs[1]
        for rumor_tweet_dir in os.listdir(os.path.join(event_dir_path, rumor_dir)):
            if rumor_tweet_dir == DS_STORE:
                continue
            source_tweet_path = os.listdir(os.path.join(os.curdir, rumor_dir, rumor_tweet_dir, 'source-tweet'))[0]
            source_tweet_path = os.path.join(event_dir_path, rumor_dir, rumor_tweet_dir, 'source-tweet',
                                             source_tweet_path)
            with open(source_tweet_path) as d:
                source_tweet = json.load(d)
            # print("source: " + str(source_tweet))
            # print(source_tweet.keys())
            source_reaction_data = {'source_tweet': source_tweet, 'reactions': []}
            reactions = os.listdir(os.path.join(os.curdir, rumor_dir, rumor_tweet_dir, 'reactions'))
            for reaction in reactions:
                reaction_path = os.path.join(event_dir_path, rumor_dir, rumor_tweet_dir, 'reactions', reaction)
                with open(reaction_path) as r:
                    reaction_tweet = json.load(r)
                    source_reaction_data['reactions'].append(reaction_tweet)
            data[event_name]['rumor_tweets'].append(source_reaction_data)
        for non_rumor_tweet_dir in os.listdir(os.path.join(event_dir_path, non_rumor_dir)):
            if non_rumor_tweet_dir == DS_STORE:
                continue
            source_tweet_path = os.listdir(os.path.join(os.curdir, non_rumor_dir, non_rumor_tweet_dir, 'source-tweet'))[
                0]
            source_tweet_path = os.path.join(event_dir_path, non_rumor_dir, non_rumor_tweet_dir, 'source-tweet',
                                             source_tweet_path)
            with open(source_tweet_path) as d:
                source_tweet = json.load(d)
            # print("source: " + str(source_tweet))
            # print(source_tweet.keys())
            source_reaction_data = {'source_tweet': source_tweet, 'reactions': []}
            reactions = os.listdir(os.path.join(os.curdir, non_rumor_dir, non_rumor_tweet_dir, 'reactions'))
            for reaction in reactions:
                reaction_path = os.path.join(event_dir_path, non_rumor_dir, non_rumor_tweet_dir, 'reactions', reaction)
                with open(reaction_path) as r:
                    reaction_tweet = json.load(r)
                    source_reaction_data['reactions'].append(reaction_tweet)
            data[event_name]['non_rumor_tweets'].append(source_reaction_data)
    print("Reading Data Done!!")
    return data


def save_to_txt():
    pheme_data = read_data_to_dir()
    name_list = list()
    name_list.append("charliehebdo")
    name_list.append("ferguson")
    name_list.append("germanwings-crash")
    name_list.append("ottawashooting")
    name_list.append("sydneysiege")

    charliehebdo = pheme_data["charliehebdo"]
    ferguson = pheme_data["ferguson"]
    germanwings_crash = pheme_data["germanwings-crash"]
    ottawasshooting = pheme_data["ottawashooting"]
    sydneysiege = pheme_data["sydneysiege"]

    event_list = list()
    event_list.append(charliehebdo)
    event_list.append(ferguson)
    event_list.append(germanwings_crash)
    event_list.append(ottawasshooting)
    event_list.append(sydneysiege)

    all_count = 0
    for event in event_list:
        success_count = 0
        actions = []
        rumors = event["rumor_tweets"]
        non_rumors = event["non_rumor_tweets"]
        current_name = name_list.pop()
        for tweets in rumors:
            # all_count = all_count + 1
            source_tweet = tweets["source_tweet"]
            source_tweet["event_name"] = current_name
            source_tweet["rumor"] = 1
            reactions = tweets["reactions"]
            document = json.dumps(source_tweet)

            actions.append(document)

            # res = es.index(index="twitter", doc_type='doc', id=source_tweet["id_str"], body=document)
            # print(res['result'])
            for tweet in reactions:
                # all_count = all_count + 1
                tweet["event_name"] = current_name
                tweet["source_tweet_id"] = source_tweet["id_str"]
                tweet["rumor"] = 1
                document = json.dumps(tweet)
                actions.append(document)

                # res = es.index(index="twitter", doc_type='doc', id=tweet["id_str"], body=document)
                # print(res['result'])
        for tweets in non_rumors:
            # all_count = all_count + 1
            source_tweet = tweets["source_tweet"]
            source_tweet["event_name"] = current_name
            source_tweet["rumor"] = 0
            document = json.dumps(source_tweet)
            actions.append(document)
            reactions = tweets["reactions"]

            # res = es.index(index="twitter", doc_type='doc', id=source_tweet["id_str"], body=document)
            # print(res['result'])
            for tweet in reactions:
                # all_count = all_count + 1
                tweet["event_name"] = current_name
                tweet["source_tweet_id"] = source_tweet["id_str"]
                tweet["rumor"] = 0
                document = json.dumps(tweet)

                actions.append(document)
                # res = es.index(index="twitter", doc_type='doc', id=tweet["id_str"], body=document)
                # print(res['result'])
        success, _ = bulk(es, actions, index="twitter", doc_type="doc", raise_on_error=True)
        success_count += success
        print("Successful Doc: " + str(success_count))
    # print(str(all_count))


save_to_txt()
