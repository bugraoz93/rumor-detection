import copy
import datetime
import json

from elasticsearch import Elasticsearch
from functools import lru_cache


class PhemeDatasetES:

    def __init__(self, hosts, index_name):
        self.es = Elasticsearch(hosts=[hosts])
        self.index_name = index_name
        self.max_lengths = dict()

    @lru_cache(maxsize=4096)
    def get_source_tweet_without_scroll(self, tweet_id):
        query = {"match": {"source_tweet_id": tweet_id}}
        print("Trying to get query: " + str(query))
        size = 1000
        data = []
        result = self.es.search(index=self.index_name, body={'size': size, 'query': query}, sort="created_at:asc")
        data.extend(map(lambda d: d['_source'], result['hits']['hits']))
        return data.copy()

    def get_data(self, query, sort):
        print("Trying to get query: " + str(query))
        size = 3000
        data = []
        if sort:
            result = self.es.search(index=self.index_name, scroll='1m', body={'size': size, 'query': query}, sort=sort)
        else:
            result = self.es.search(index=self.index_name, scroll='1m', body={'size': size, 'query': query})
        total_count = int(result['hits']['total'])
        data.extend(map(lambda d: d['_source'], result['hits']['hits']))
        scroll_id = result['_scroll_id']
        while len(data) < total_count:
            PhemeDatasetES._print_progress(len(data), total_count)
            result = self.es.scroll(scroll_id=scroll_id, scroll="1m")
            data.extend(map(lambda d: d['_source'], result['hits']['hits']))
            scroll_id = result['_scroll_id']

        return data

    def get_data_event_name(self, event_name):
        print("Starting to fetch event: " + str(event_name))
        return self.get_data({'match': {'event_name': event_name}}, None)

    def get_event_time_frames(self, event_name, frame_count):
        result = self.get_data({'match': {'event_name': event_name}}, 'created_at:asc')
        first = self.get_timestamp(result[0])
        last = self.get_timestamp(result[-1])
        frame_size = int((last - first) / frame_count)
        counter = 1
        time_frames = []
        current_frame = []
        length = 0
        for data in result:
            if self.get_timestamp(data) < (first + (counter * frame_size)):
                current_frame.append(data)
                length += 1
            else:
                print(str(counter) + " Frame Size: " + str(len(current_frame)))
                time_frames.append(current_frame)
                current_frame = []
                counter += 1
        print("Length: " + str(length))
        print("Result length: " + str(len(result)))
        return time_frames

    def get_event_time_frames_with_time(self, cij, frame_size):
        result = cij
        first = self.get_timestamp(result[0])
        counter = 1
        time_frames = []
        current_frame = []
        length = 0
        for data in result:
            if self.get_timestamp(data) < (first + (counter * frame_size)):
                current_frame.append(data)
                length += 1
            else:
                # print(str(counter) + " Frame Size: " + str(len(current_frame)))
                time_frames.append(current_frame)
                counter += 1
                while self.get_timestamp(data) >= (first + (counter * frame_size)):
                    current_frame = []
                    time_frames.append(current_frame.copy())
                    counter += 1
                    # print(str(counter) + " Frame Size: " + str(len(current_frame)))
                current_frame.append(data)
                length += 1

        time_frames.append(current_frame.copy())
        # print(str(counter) + " Frame Size: " + str(len(current_frame)))
        # print("Time Frame length: " + str(len(time_frames)))
        return time_frames

    @lru_cache(maxsize=4096)
    def get_source_and_reactions(self, event_name):
        source_tweets = self.get_data({'bool': {'must': [{'match': {'event_name': event_name}}],
                                                'must_not': [{'exists': {'field': 'source_tweet_id'}}]}},
                                      'created_at:asc')
        conversations = list()
        for source_tweet in source_tweets:
            conversation = dict()
            conversation["source_tweet"] = source_tweet
            conversation["reactions"] = self.get_source_tweet_without_scroll(source_tweet["id_str"])
            conversations.append(conversation.copy())

        return conversations

    def get_all_cij(self, event_name, t):
        conversations = self.get_source_and_reactions(event_name)
        frames = list()
        rumors = list()
        representations = list()
        for cij in conversations:
            pre_frame = list()
            pre_frame.append(cij["source_tweet"])
            rumors.append(cij["source_tweet"]["rumor"])
            pre_frame.extend(cij["reactions"])
            representations.append(self.get_source_tweet_representation(cij["source_tweet"]))
            frame = self.get_event_time_frames_with_time(pre_frame, t)
            frames.append(frame.copy())

        return frames, rumors, representations

    def get_vectors_of_cij(self, event_name, t):
        vectors = list()
        frames, rumors, representations = self.get_all_cij(event_name, t)
        for frame in frames:
            vector = list()
            for cij in frame:
                vector.append(len(cij))
            vectors.append(vector.copy())

        return vectors, rumors, representations

    def get_vectors_of_cij_with_max_lengths(self, event_name, t):
        print("Vectors creating for event " + str(event_name) + " and for the time " + str(t / 60))
        vectors, rumors, representations = self.get_vectors_of_cij(event_name, t)
        current_max_length = max(len(x) for x in vectors)
        current_key = str(int(t / 60))
        if current_key in self.max_lengths:
            if current_max_length > self.max_lengths[current_key]:
                self.max_lengths[current_key] = max(len(x) for x in vectors)
        else:
            self.max_lengths[current_key] = max(len(x) for x in vectors)

        return vectors, rumors, representations

    def get_vectors_of_cij_with_padding_only_event(self, event_name):
        times = {"2": 2*60, "5": 5*60, "10": 10*60, "30": 30*60, "60": 60*60}
        vectors_times = dict()
        vectors, rumors, representations = self.get_vectors_of_cij_with_max_lengths(event_name, 2*60)
        for time in times.keys():
            vectors_times[time], _, _ = self.get_vectors_of_cij_with_max_lengths(event_name, times[time])

        vectors_times_copy = copy.deepcopy(vectors_times)

        combined_features = list()
        for i in range(len(vectors)):
            combined_feature = dict()
            for vector_time_key in vectors_times.keys():
                combined_feature["vector-" + str(vector_time_key)] = vectors_times[vector_time_key].pop(0)
            current_representation = representations.pop(0)
            for representation in current_representation.keys():
                combined_feature[representation] = current_representation[representation]
            combined_feature["rumor"] = rumors.pop(0)
            combined_features.append(combined_feature.copy())

        return combined_features, vectors_times_copy

    def write_combined_features_to_file(self):
        events = ["charliehebdo", "germanwings-crash", "sydneysiege", "ottawashooting", "ferguson"]

        combined_features = dict()
        combined_features_vector_times = dict()
        for event in events:
            combined_features[event], combined_features_vector_times = self.get_vectors_of_cij_with_padding_only_event(event)

        for event in events:
            for vector_combined in combined_features[event]:
                for vectors_time_key in combined_features_vector_times.keys():
                    current_max_length = self.max_lengths[vectors_time_key]
                    vector = vector_combined["vector-" + str(vectors_time_key)]
                    if len(vector) < current_max_length:
                        for i in range(current_max_length - len(vector)):
                            vector.append(0)

            with open(str("files/" + event + ".txt"), "w") as file:
                for combined_feature in combined_features[event]:
                    file.write(json.dumps(combined_feature) + "\n")
            file.close()

    @staticmethod
    def read_combined_features_from_file(event_name):
        with open(str("files/" + event_name + ".txt"), "r") as file:
            combined_features_lines = (file.readlines())
        combined_features = list()
        for line in combined_features_lines:
            combined_features.append(json.loads(line))
        file.close()
        return combined_features

    @staticmethod
    def get_speed_of_time_frame(time_frame):
        first_time = PhemeDatasetES.get_timestamp(time_frame[0]["created_at"])
        last_time = PhemeDatasetES.get_timestamp(time_frame[-1]["created_at"])
        total_time = last_time - first_time

        return len(time_frame) / int(total_time)

    @staticmethod
    def get_speed_of_time_frame_replies(time_frame):
        for item in time_frame:
            if "source_tweet_id" not in item:
                time_frame.remove(item)

        return PhemeDatasetES.get_speed_of_time_frame(time_frame)

    @staticmethod
    def get_retweet_count_of_source(time_frame):
        ave_retweet_count_of_time_frame = sum([int(item["retweet_count"]) for item in time_frame])
        if ave_retweet_count_of_time_frame != 0:
            time_frame_ave = int(ave_retweet_count_of_time_frame) / len(time_frame)
        else:
            return None
        source_tweet_retweet_normalized = list()
        for item in time_frame:
            if "source_tweet_id" not in item:
                item["retweet_count_normalized"] = item["retweet_count"] / time_frame_ave
                source_tweet_retweet_normalized.append(item)

        return source_tweet_retweet_normalized

    @staticmethod
    def get_timestamp(data):
        return int(datetime.datetime.strptime(data["created_at"], '%a %b %d %H:%M:%S +0000 %Y').timestamp())

    @staticmethod
    def get_timestamp_of_user(data):
        return int(datetime.datetime.strptime(data['user']["created_at"], '%a %b %d %H:%M:%S +0000 %Y').timestamp())

    @staticmethod
    def _print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()

    @staticmethod
    def __get_total_time_of_time_frame(time_frame):
        start = PhemeDatasetES.get_timestamp(time_frame[0])
        end = PhemeDatasetES.get_timestamp(time_frame[-1])
        return end - start

    @staticmethod
    def __get_unique_user_count(time_frame):
        users = set()
        for post in time_frame:
            users.add(post['user']['id_str'])

        return len(users)

    @staticmethod
    def __get_average_user_per_second(time_frame):
        user_count = PhemeDatasetES.__get_unique_user_count(time_frame)
        total_time = PhemeDatasetES.__get_total_time_of_time_frame(time_frame)
        return user_count / total_time

    @staticmethod
    def __get_average_default_user_per_second(time_frame):
        total_time = PhemeDatasetES.__get_total_time_of_time_frame(time_frame)
        filtered_frame = list(filter(lambda x: x['user']['default_profile'] is True, time_frame))
        return len(filtered_frame) / total_time

    def __get_total_reaction_time(self, source_tweet):
        response_data = self.get_source_tweet_without_scroll(source_tweet['id_str'])
        if len(response_data) > 0:
            start = self.get_timestamp(response_data[0])
            end = self.get_timestamp(response_data[-1])
            return end - start
        else:
            return 0

    def __get_total_reaction_count(self, source_tweet, max_time):
        response_data = self.get_source_tweet_without_scroll(source_tweet['id_str'])
        if max_time is None:
            return len(response_data)
        if len(response_data) == 0:
            return 0
        start = self.get_timestamp(response_data[0])
        max_stamp = start + max_time
        return len(list(filter(lambda x: self.get_timestamp(x) < max_stamp, response_data)))

    def __get_total_reaction_mention_count(self, source_tweet):
        response_data = self.get_source_tweet_without_scroll(source_tweet['id_str'])
        if len(response_data) == 0:
            return 0
        total_user_mention_count = 0
        for tweet in response_data:
            if 'entities' in tweet and 'user_mentions' in tweet['entities']:
                total_user_mention_count += len(tweet['entities']['user_mentions'])
        return total_user_mention_count

    def __get_total_reaction_retweet_count(self, source_tweet):
        response_data = self.get_source_tweet_without_scroll(source_tweet['id_str'])
        if len(response_data) == 0:
            return 0
        total_retweet_count = 0
        for tweet in response_data:
            total_retweet_count += int(tweet['retweet_count'])
        return total_retweet_count

    def __get_user_is_geo_enabled(self, source_tweet):
        response_data = self.get_source_tweet_without_scroll(source_tweet["id_str"])
        if len(response_data) == 0:
            return 0
        else:
            return int(response_data[0]["user"]["geo_enabled"] == "true")

    def __get_user_has_description(self, source_tweet):
        response_data = self.get_source_tweet_without_scroll(source_tweet["id_str"])
        if len(response_data) == 0:
            return 0
        else:
            return int(len(response_data[0]["user"]["description"]) > 0)

    def __get_user_description_count(self, source_tweet):
        response_data = self.get_source_tweet_without_scroll(source_tweet["id_str"])
        if len(response_data) == 0:
            return 0
        else:
            if len(response_data[0]["user"]["description"]) > 0:
                return len(response_data[0]["user"]["description"].split(' '))
            else:
                return 0

    @staticmethod
    def __get_has_question_mark(text):
        if '?' in text:
            return 1
        else:
            return 0

    @staticmethod
    def __get_number_of_question_mark(text):
        if '?' in text:
            count = 0
            for i in text:
                if i == '?':
                    count = count + 1
            return count
        else:
            return 0

    @staticmethod
    def __get_has_exclamation_mark(text):
        if '!' in text:
            return 1
        else:
            return 0

    @staticmethod
    def __get_number_of_exclamation_mark(text):
        if '!' in text:
            count = 0
            for i in text:
                if i == '!':
                    count = count + 1
            return count
        else:
            return 0

    @staticmethod
    def __get_has_dotdotdot(text):
        if '...' in text:
            return 1
        else:
            return 0

    @staticmethod
    def __get_number_of_dotdotdot(text):
        if '...' in text:
            count = 0
            for i in text:
                if i == '...':
                    count = count + 1
            return count
        else:
            return 0

    def get_source_tweet_representation(self, source_tweet):
        total_reaction_count = self.__get_total_reaction_count(source_tweet, None)
        total_time_span = self.__get_total_reaction_time(source_tweet)
        if total_time_span == 0:
            total_time_span = 1
        if source_tweet["user"]["friends_count"] > 0:
            role_score = source_tweet['user']['followers_count'] / source_tweet['user']['friends_count']
        else:
            role_score = 0
        return {
                # 'id': source_tweet['id_str'],
                # 'isRumor': source_tweet['rumor'] == 1,
                'time_span': total_time_span,  # deeper
                'is_sensitive': int(
                    'possibly_sensitive' in source_tweet and source_tweet['possibly_sensitive'] is True),
                'first_five_reaction_count': self.__get_total_reaction_count(source_tweet, 5 * 60),
                'early_reaction_count': self.__get_total_reaction_count(source_tweet, 15 * 60),
                'mid_reaction_count': self.__get_total_reaction_count(source_tweet, 30 * 60),
                'late_reaction_count': self.__get_total_reaction_count(source_tweet, 60 * 60),
                'all_reaction_count': self.__get_total_reaction_count(source_tweet, None),
                'media_count': len(
                    (source_tweet['entities'] and 'media' in source_tweet['entities']) and source_tweet['entities'][
                        'media'] or []),
                'hashtag_count': len(
                    source_tweet['entities'] and ('hashtags' in source_tweet['entities']) and source_tweet['entities'][
                        'hashtags'] or []),
                # User Features
                'is_geo_enabled': self.__get_user_is_geo_enabled(source_tweet),
                'has_description': self.__get_user_has_description(source_tweet),
                'description_word_count': self.__get_user_description_count(source_tweet),
                'role_score': int(role_score),
                'user_follower_count': source_tweet['user']['followers_count'],
                'is_verified': int(source_tweet['user']['verified'] is True),
                'favorites_count': source_tweet['user']['favourites_count'],
                'engagement_score': (source_tweet['user']['statuses_count']) /
                                    (datetime.datetime.now().timestamp() - self.get_timestamp_of_user(source_tweet)),
                # Tweet Features
                'has_question_mark': self.__get_has_question_mark(source_tweet["text"]),
                'question_mark_count': self.__get_number_of_question_mark(source_tweet["text"]),
                'has_exclamation_mark': self.__get_has_exclamation_mark(source_tweet["text"]),
                'exclamation_mark_count': self.__get_number_of_exclamation_mark(source_tweet["text"]),
                'has_dotdotdot_mark': self.__get_has_dotdotdot(source_tweet["text"]),
                'dotdotdot_mark_count': self.__get_number_of_dotdotdot(source_tweet["text"]),

                'reaction_speed': total_reaction_count / total_time_span,  # faster
                'reaction_mention_count': self.__get_total_reaction_mention_count(source_tweet),
                'reaction_retweet_count': self.__get_total_reaction_retweet_count(source_tweet),  # broader
                'user_event_time_diff': int(self.get_timestamp(source_tweet) - self.get_timestamp_of_user(source_tweet))
                }

    def get_source_tweet_representations(self, event_name):
        data = self.get_data_event_name(event_name)
        features = []

        tweets = filter(lambda x: 'source_tweet_id' not in x, data)
        for source_tweet in tweets:
            total_reaction_count = self.__get_total_reaction_count(source_tweet, None)
            total_time_span = self.__get_total_reaction_time(source_tweet)
            if total_time_span == 0:
                total_time_span = 1
            if source_tweet["user"]["friends_count"] > 0:
                role_score = source_tweet['user']['followers_count'] / (source_tweet['user']['friends_count'])
            else:
                role_score = 0
            features.append(
                {'id': source_tweet['id_str'],
                 'isRumor': source_tweet['rumor'] == 1,
                 'time_span': total_time_span,  # deeper
                 'is_sensitive': int(
                     'possibly_sensitive' in source_tweet and source_tweet['possibly_sensitive'] is True),
                 'first_five_reaction_count': self.__get_total_reaction_count(source_tweet, 5 * 60),
                 'early_reaction_count': self.__get_total_reaction_count(source_tweet, 15 * 60),
                 'mid_reaction_count': self.__get_total_reaction_count(source_tweet, 30 * 60),
                 'late_reaction_count': self.__get_total_reaction_count(source_tweet, 60 * 60),
                 'all_reaction_count': self.__get_total_reaction_count(source_tweet, None),
                 'media_count': len(
                     (source_tweet['entities'] and 'media' in source_tweet['entities']) and source_tweet['entities'][
                         'media'] or []),
                 'hashtag_count': len(
                     source_tweet['entities'] and ('hashtags' in source_tweet['entities']) and source_tweet['entities'][
                         'hashtags'] or []),
                 # User Features
                 'is_geo_enabled': self.__get_user_is_geo_enabled(source_tweet),
                 'has_description': self.__get_user_has_description(source_tweet),
                 'description_word_count': self.__get_user_description_count(source_tweet),
                 'role_score': int(role_score),
                 'user_follower_count': source_tweet['user']['followers_count'],
                 'is_verified': int(source_tweet['user']['verified'] is True),
                 'favorites_count': source_tweet['user']['favourites_count'],
                 'engagement_score': (source_tweet['user']['statuses_count']) /
                                     (datetime.datetime.now().timestamp() - self.get_timestamp_of_user(source_tweet)),
                 # Tweet Features
                 'has_question_mark': self.__get_has_question_mark(source_tweet["text"]),
                 'question_mark_count': self.__get_number_of_question_mark(source_tweet["text"]),
                 'has_exclamation_mark': self.__get_has_exclamation_mark(source_tweet["text"]),
                 'exclamation_mark_count': self.__get_number_of_exclamation_mark(source_tweet["text"]),
                 'has_dotdotdot_mark': self.__get_has_dotdotdot(source_tweet["text"]),
                 'dotdotdot_mark_count': self.__get_number_of_dotdotdot(source_tweet["text"]),

                 'reaction_speed': total_reaction_count / total_time_span,  # faster
                 'reaction_mention_count': self.__get_total_reaction_mention_count(source_tweet),
                 'reaction_retweet_count': self.__get_total_reaction_retweet_count(source_tweet),  # broader
                 'user_event_time_diff': int(
                     self.get_timestamp(source_tweet) - self.get_timestamp_of_user(source_tweet))
                 }
            )
        return features
