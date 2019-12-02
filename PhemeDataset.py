import datetime

from elasticsearch import Elasticsearch


class PhemeDatasetES:

    def __init__(self, hosts, index_name):
        self.es = Elasticsearch(hosts=[hosts])
        self.index_name = index_name

    def get_data(self, query, sort):
        size = 1000
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

    def get_event_time_frames_with_time(self, event_name, frame_size):
        result = self.get_data({'match': {'event_name': event_name}}, 'created_at:asc')
        first = self.get_timestamp(result[0])
        last = self.get_timestamp(result[-1])
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
                counter += 1
                while self.get_timestamp(data) >= (first + (counter * frame_size)):
                    current_frame = []
                    time_frames.append(current_frame.copy())
                    counter += 1
                    print(str(counter) + " Frame Size: " + str(len(current_frame)))
                current_frame.append(data)
                length += 1

        time_frames.append(current_frame.copy())
        print(str(counter) + " Frame Size: " + str(len(current_frame)))
        print("Time Frame length: " + str(len(time_frames)))
        return time_frames

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
        query = {"match": {"source_tweet_id": source_tweet['id_str']}}
        response_data = self.get_data(query, "created_at:asc")
        if len(response_data) > 0:
            start = self.get_timestamp(response_data[0])
            end = self.get_timestamp(response_data[-1])
            return end - start
        else:
            return 0

    def __get_total_reaction_count(self, source_tweet, max_time):
        query = {"match": {"source_tweet_id": source_tweet['id_str']}}
        response_data = self.get_data(query, "created_at:asc")
        if max_time is None:
            return len(response_data)
        if len(response_data) == 0:
            return 0
        start = self.get_timestamp(response_data[0])
        max_stamp = start + max_time
        return len(list(filter(lambda x: self.get_timestamp(x) < max_stamp, response_data)))

    def get_source_tweet_representations(self, event_name):
        data = self.get_data_event_name(event_name)
        features = []
        for source_tweet in filter(lambda x: 'source_tweet_id' not in x, data):
            features.append(
                {'id': source_tweet['id_str'],
                 'isRumor': source_tweet['rumor'] == 1,
                 'time_span': self.__get_total_reaction_time(source_tweet),
                 'early_reaction_count': self.__get_total_reaction_count(source_tweet, 15*60),
                 'mid_reaction_count': self.__get_total_reaction_count(source_tweet, 60*60),
                 'all_reaction_count': self.__get_total_reaction_count(source_tweet, None)
                 }
            )
        return features
