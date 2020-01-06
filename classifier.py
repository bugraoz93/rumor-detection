import json
from PhemeDataset import PhemeDatasetES
from sklearn.metrics import classification_report, precision_score, recall_score, mean_squared_error, accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.naive_bayes import MultinomialNB

dataset = PhemeDatasetES(hosts="http://localhost:9200", index_name="pheme_tweet_data")

EVENTS = [
    'charliehebdo',
    'germanwings-crash',
    'sydneysiege',
    'ottawashooting',
    'ferguson'
]

TIME_FRAME_SIZES = [  # In terms of minutes
    2,
    5,
    10,
    30,
    60,
]

SELECTED_FEATURES = [
    'engagement_score',
    'role_score',
    'user_follower_count',
    'reaction_speed'
]



class BasicVoter:

    def fit(self, X, y):
        pass

    def predict(self, X):
        y = []
        for x in X:
            if sum(x) > int(len(x) / 2):
                y.append(1)
            else:
                y.append(0)
        return y

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

class EventSelector:

    def __init__(self):
        self.events = EVENTS
        self.test_event = 0

    def get_one_partition(self):
        for partition in self.__iter__():
            return partition

    def __iter__(self):
        while True:
            if self.test_event == len(self.events):
                break
            yield [self.events[i] for i in range(len(self.events)) if i != self.test_event], \
                  self.events[self.test_event]
            self.test_event = (self.test_event + 1)


class RumorClassifier:

    def __init__(self):
        self.representations = {}
        for e in EVENTS:
            self.representations[e] = dataset.read_combined_features_from_file(e)

        for s in SELECTED_FEATURES:
            max_score = 0
            min_score = 999999
            for e in EVENTS:

                    for rep in self.representations[e]:
                        if rep[s] > max_score:
                            max_score = rep[s]
                        if rep[s] < min_score:
                            min_score = rep[s]
            for e in EVENTS:
                for rep in self.representations[e]:
                    rep[s] = (rep[s] - min_score)\
                                              / (max_score - min_score)

    def __representation_to_prop_level1_features(self, event_name, t):
        X = []
        y = []
        for rep in self.representations[event_name]:
            X.append(rep['vector-' + str(t)])
            y.append(rep['rumor'])

        return X, y

    def __representation_to_level2_features(self, event_name, props):
        X = []
        y = []
        for i in range(len(self.representations[event_name])):
            X.extend([*list(self.representations[event_name][i]['features'].values()), *props[i]])
            y.append(self.representations[event_name][i]['rumor'])

        return X, y

    def correct_padding(self, X1, X2):
        max_len = max(len(X1[0]), len(X2[0]))
        if len(X2[0]) == max_len:
            for i in range(len(X1)):
                X1[i].extend([0 for _ in range(max_len - len(X1[i]))])
        else:
            for i in range(len(X2)):
                X2[i].extend([0 for _ in range(max_len - len(X2[i]))])
        return X1,X2

    def train_and_evaluate(self):
        prob_models = {}
        e = EventSelector()

        results = {

        }

        for train_events, test_event in e:

            results[test_event] = {}

            for t_size in TIME_FRAME_SIZES:
                model = MultinomialNB()
                model_train_data_X = []
                model_train_data_y = []

                model_test_data_X, model_test_data_y = self.__representation_to_prop_level1_features(test_event, t_size)
                for event in train_events:
                    X, y = self.__representation_to_prop_level1_features(event, t_size)
                    model_train_data_X.extend(X)
                    model_train_data_y.extend(y)


                model.fit(model_train_data_X, model_train_data_y)

                y_pred_train = model.predict_proba(model_train_data_X)[:, 1]
                #y_pred_train = model.predict(model_train_data_X)
                train_loss = mean_squared_error(model_train_data_y, y_pred_train)
                print("Train Loss %s: %f" % (str(t_size), train_loss))

                y_pred = model.predict_proba(model_test_data_X)[:, 1]
                #y_pred = model.predict(model_test_data_X)
                test_loss = mean_squared_error(model_test_data_y, y_pred)
                print("Test Loss %s:  %f" % (str(t_size),  test_loss))

                prob_models[str(t_size)] = {}
                prob_models[str(t_size)]['model'] = model
                prob_models[str(t_size)]['preds_train'] = y_pred_train
                prob_models[str(t_size)]['preds_test'] = y_pred

            level2_train_X = []
            level2_train_y = []
            level2_test_X = []
            level2_test_y = []

            for e in train_events:
                for i in range(len(self.representations[e])):
                    X_train = []
                    for t_size in TIME_FRAME_SIZES:
                        X_train.append(prob_models[str(t_size)]['preds_train'][i])
                    X_train.extend([v for k, v in self.representations[e][i].items() if k != 'rumor' and k in SELECTED_FEATURES])
                    level2_train_X.append(X_train)
                    level2_train_y.append(self.representations[e][i]['rumor'])

            for i in range(len(self.representations[test_event])):
                X_test = []
                for t_size in TIME_FRAME_SIZES:
                    X_test.append(prob_models[str(t_size)]['preds_test'][i])
                X_test.extend([v for k, v in self.representations[test_event][i].items() if k != 'rumor' and k in SELECTED_FEATURES])
                level2_test_X.append(X_test)
                level2_test_y.append(self.representations[test_event][i]['rumor'])

            c_0 = len([y for y in level2_train_y if y == 0])
            c_1 = len(level2_train_y) - c_0
            class_weights = {0: 1, 1: c_0/c_1}

            decision_tree = DecisionTreeClassifier(max_depth=8)
            decision_tree.fit(level2_train_X, level2_train_y)
            level2_pred_test = decision_tree.predict(level2_test_X)
            level2_pred_train = decision_tree.predict(level2_train_X)

            c_train = classification_report(level2_train_y, level2_pred_train)
            c_test = classification_report(level2_test_y, level2_pred_test)
            precision = precision_score(level2_test_y, level2_pred_test)
            recall = recall_score(level2_test_y, level2_pred_test)
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = accuracy_score(level2_test_y, level2_pred_test)
            balanced_accuracy = balanced_accuracy_score(level2_test_y, level2_pred_test)
            results[test_event]['decision_tree'] = {'precision': precision,
                                                    'recall': recall,
                                                    'F1': f1,
                                                    'accuracy': accuracy,
                                                    'balanced_accuracy': balanced_accuracy
                                                    }
            print("*-*-*-*-*-DT Classificiation Results TEST (" + test_event + ")-*-*-*-*-*-*-*-*")
            print(c_test)


            mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1200)
            mlp.fit(level2_train_X, level2_train_y)
            level2_pred_test = mlp.predict(level2_test_X)
            level2_pred_train = mlp.predict(level2_train_X)

            c_test = classification_report(level2_test_y, level2_pred_test)
            precision = precision_score(level2_test_y, level2_pred_test)
            recall = recall_score(level2_test_y, level2_pred_test)
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = accuracy_score(level2_test_y, level2_pred_test)
            balanced_accuracy = balanced_accuracy_score(level2_test_y, level2_pred_test)
            results[test_event]['MLP'] = {'precision': precision, 'recall': recall, 'F1': f1,
                                          'accuracy': accuracy,
                                          'balanced_accuracy': balanced_accuracy
                                          }
            print("*-*-*-*-*-MLP Classificiation Results TEST (" + test_event + ")-*-*-*-*-*-*-*-*")
            print(c_test)


            svc = SVC(C = 10.0)
            svc.fit(level2_train_X, level2_train_y)
            level2_pred_test = svc.predict(level2_test_X)
            level2_pred_train = svc.predict(level2_train_X)

            c_test = classification_report(level2_test_y, level2_pred_test)
            precision = precision_score(level2_test_y, level2_pred_test)
            recall = recall_score(level2_test_y, level2_pred_test)
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = accuracy_score(level2_test_y, level2_pred_test)
            balanced_accuracy = balanced_accuracy_score(level2_test_y, level2_pred_test)
            results[test_event]['SVM'] = {'precision': precision, 'recall': recall, 'F1': f1,
                                          'accuracy': accuracy,
                                          'balanced_accuracy': balanced_accuracy
                                          }
            print("*-*-*-*-*-SVM Classificiation Results TEST (" + test_event + ")-*-*-*-*-*-*-*-*")
            print(c_test)

            rf = RandomForestClassifier(n_estimators=150, max_depth=10)
            rf.fit(level2_train_X, level2_train_y)
            level2_pred_test = rf.predict(level2_test_X)
            level2_pred_train = rf.predict(level2_train_X)

            c_test = classification_report(level2_test_y, level2_pred_test)
            precision = precision_score(level2_test_y, level2_pred_test)
            recall = recall_score(level2_test_y, level2_pred_test)
            f1 = 2 * precision * recall / (precision + recall)
            accuracy = accuracy_score(level2_test_y, level2_pred_test)
            balanced_accuracy = balanced_accuracy_score(level2_test_y, level2_pred_test)
            results[test_event]['RF'] = {'precision': precision, 'recall': recall, 'F1': f1,
                                         'accuracy': accuracy,
                                         'balanced_accuracy': balanced_accuracy
                                         }
            print("*-*-*-*-*-RF Classificiation Results TEST (" + test_event + ")-*-*-*-*-*-*-*-*")
            print(c_test)

            km = KMeans(n_clusters=2)
            km.fit(level2_train_X, level2_train_y)
            level2_pred_test = km.predict(level2_test_X)
            level2_pred_train = km.predict(level2_train_X)

            c_test = classification_report(level2_test_y, level2_pred_test)
            precision = precision_score(level2_test_y, level2_pred_test)
            recall = recall_score(level2_test_y, level2_pred_test)
            accuracy = accuracy_score(level2_test_y, level2_pred_test)
            balanced_accuracy = balanced_accuracy_score(level2_test_y, level2_pred_test)
            f1 = 2 * precision * recall / (precision + recall)
            results[test_event]['KMeans'] = {'precision': precision, 'recall': recall, 'F1': f1,
                                             'accuracy': accuracy,
                                             'balanced_accuracy': balanced_accuracy
                                             }
            print("*-*-*-*-*-Kmeans Classificiation Results TEST (" + test_event + ")-*-*-*-*-*-*-*-*")
            print(c_test)

        return results


c = RumorClassifier()
results = c.train_and_evaluate()


print(json.dumps(results, indent=2))


print("Bitti")

