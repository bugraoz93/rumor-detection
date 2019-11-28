import os
import json

dataset_directory = "/home/cluster/PycharmProjects/rumor-detection/data"
DS_STORE = '.DS_Store'
directories = os.listdir(dataset_directory)
directories.remove('README')
directories.remove(DS_STORE)


def read_data_to_dir():
    data = {}
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
            print("source: " + str(source_tweet))
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
            print("source: " + str(source_tweet))
            source_reaction_data = {'source_tweet': source_tweet, 'reactions': []}
            reactions = os.listdir(os.path.join(os.curdir, non_rumor_dir, non_rumor_tweet_dir, 'reactions'))
            for reaction in reactions:
                reaction_path = os.path.join(event_dir_path, non_rumor_dir, non_rumor_tweet_dir, 'reactions', reaction)
                with open(reaction_path) as r:
                    reaction_tweet = json.load(r)
                    source_reaction_data['reactions'].append(reaction_tweet)
            data[event_name]['non_rumor_tweets'].append(source_reaction_data)

    return data


pheme_data = read_data_to_dir()