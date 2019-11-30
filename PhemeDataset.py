from elasticsearch import Elasticsearch


class PhemeDatasetES:

    def __init__(self):
        self.es = Elasticsearch(hosts="0.0.0.0:9200")
        self.index_name = 'pheme_tweet_data'

    def get_data_with_event_name(self, event_name):
        size = 1000
        data = []
        result = self.es.search(index=self.index_name, scroll='1m', body={'size': size, 'query': {'match': {'event_name': event_name}}})
        total_count = int(result['hits']['total'])
        data.extend(map(lambda d: d['_source'], result['hits']['hits']))
        scroll_id = result['_scroll_id']
        while len(data) < total_count:
            PhemeDatasetES._print_progress(len(data), total_count)
            result = self.es.scroll(scroll_id=scroll_id, scroll="1m")
            data.extend(map(lambda d: d['_source'], result['hits']['hits']))
            scroll_id = result['_scroll_id']

        return data

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


