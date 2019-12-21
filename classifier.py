from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from PhemeDataset import PhemeDatasetES
from sklearn.metrics import classification_report, precision_score, recall_score

dataset = PhemeDatasetES(hosts="http://0.0.0.0:9200", index_name="pheme_tweet_data")


def get_all_data(dataset):
    representations = []
    representations.extend(dataset.get_source_tweet_representations("charliehebdo"))
    representations.extend(dataset.get_source_tweet_representations("germanwings-crash"))
    representations.extend(dataset.get_source_tweet_representations("sydneysiege"))
    representations.extend(dataset.get_source_tweet_representations("ottawashooting"))
    representations.extend(dataset.get_source_tweet_representations("ferguson"))
    return representations

features = ['time_span',
            'early_reaction_count',
             'mid_reaction_count',
             'all_reaction_count',
             'first_five_reaction_count',
             'late_reaction_count',
             'media_count',
             'hashtag_count',
             'reaction_speed',
             'reaction_mention_count',
             'reaction_retweet_count',
             'user_event_time_diff',
             # User
             'is_geo_enabled',
             'has_description',
             'description_word_count',
             'role_score',
             'user_follower_count',
             'is_verified',
             'favorites_count',
             'engagement_score',
             # Tweet
             'has_question_mark',
             'question_mark_count',
             'has_exclamation_mark',
             'exclamation_mark_count',
             'has_dotdotdot_mark',
             'dotdotdot_mark_count'
            ]


def get_train_test_set(dataset):
    X = []
    y = []
    all_data = get_all_data(dataset)
    for representation in all_data:
        x = [representation['time_span'],
             representation['early_reaction_count'],
             representation['mid_reaction_count'],
             representation['all_reaction_count'],
             representation['first_five_reaction_count'],
             representation['late_reaction_count'],
             representation['media_count'],
             representation['hashtag_count'],
             representation['reaction_speed'],
             representation['reaction_mention_count'],
             representation['reaction_retweet_count'],
             representation['user_event_time_diff'],
             # User
             representation['is_geo_enabled'],
             representation['has_description'],
             representation['description_word_count'],
             representation['role_score'],
             representation['user_follower_count'],
             representation['is_verified'],
             representation['favorites_count'],
             representation['engagement_score'],
             # Tweet
             representation['has_question_mark'],
             representation['question_mark_count'],
             representation['has_exclamation_mark'],
             representation['exclamation_mark_count'],
             representation['has_dotdotdot_mark'],
             representation['dotdotdot_mark_count']

             ]
        X.append(x.copy())
        y.append(representation['isRumor'])

    return X, y, all_data


def eleminate_features(X, y, select_k_best=10):
    # feature_selection, p = f_classif(X, y)
    selector = SelectKBest(f_classif, k=select_k_best)
    X_new = selector.fit_transform(X, y)

    # threshold = sum([a for a in p if a > 0]) / len(p)
    # sorted_p = sorted(p, reverse=True)
    # threshold = sorted_p[select_k_best]
    # X_new = []
    # for x in X:
    #     selected_indices = [j for j in range(len(x)) if p[j] > threshold]
    #     X_new.append(list([x[k] for k in selected_indices]))
    y_new = y.copy()
    for j in range(len(y_new)):
        if y_new[j]:
            y_new[j] = 1
        else:
            y_new[j] = 0

    return X_new, y_new, selector.scores_

def eleminate_features_unvariant(X, y, select_k_best=10):
    feature_selection, p = f_classif(X, y)
    # threshold = sum([a for a in p if a > 0]) / len(p)
    sorted_p = sorted(p, reverse=True)
    threshold = sorted_p[select_k_best]

    X_new = []
    for x in X:
        selected_indices = [j for j in range(len(x)) if p[j] > threshold]
        X_new.append(list([x[k] for k in selected_indices]))
    y_new = y.copy()
    for j in range(len(y_new)):
        if y_new[j]:
            y_new[j] = 1
        else:
            y_new[j] = 0

    return X_new, y_new


def parameter_selection(X_train, y_train):
    min_sample_split = [5, 10, 25]
    max_features = [x for x in range(4, len(X_train[0]), 2)]
    max_depth = [20, 25, 30, 35]
    best_param = {'msp': -1, 'md': -1, 'mf': -1, 'mf1': 0}
    max_f1 = 0
    for msp in min_sample_split:
        for mf in max_features:
            for md in max_depth:
                rfc = RandomForestClassifier(n_estimators=100,
                                             max_depth=md,
                                             min_samples_split=msp,
                                             max_features=mf)
                split_count = 5
                kf = KFold(n_splits=split_count)
                f1 = 0
                for train_index, validate_index in kf.split(X_train):

                    rfc.fit([X_train[i] for i in train_index], [y_train[i] for i in train_index])
                    y_predict = rfc.predict([X_train[i] for i in validate_index])
                    r = recall_score([y_train[i] for i in validate_index], y_predict)
                    p = precision_score([y_train[i] for i in validate_index], y_predict)
                    if (p + r) > 0.01:
                        f1 += float(2 * p * r) / float(p + r)

                f1 = f1 / split_count
                print("F1:" + str(f1))

                if f1 > max_f1:
                    best_param['mf1'] = f1
                    best_param['msp'] = msp
                    best_param['md'] = md
                    best_param['mf'] = mf
                    max_f1 = f1

    return best_param


def train_and_evaluate_rfc(X_train, X_test, y_train, y_test, params):
    rfc = RandomForestClassifier(n_estimators=250,
                                 max_depth=params['md'],
                                 min_samples_split=params['msp'],
                                 max_features=params['mf'])

    rfc.fit(X_train, y_train)
    y_predict = rfc.predict(X_test)
    y_predict_train = rfc.predict(X_train)
    c_test = classification_report(y_test, y_predict)
    c_train = classification_report(y_train, y_predict_train)
    r = recall_score(y_test, y_predict)
    p = precision_score(y_test, y_predict)

    print("Precision: " + str(p))
    print("Recall:" + str(r))
    print("-*-*-*-Classification Report Test-*-*-*-*-")
    print(c_test)
    print("-*-*-*-Classification Report Train-*-*-*-*-")
    print(c_train)


def train_and_evaluate_gbc(X_train, X_test, y_train, y_test, params):
    gbc = GradientBoostingClassifier(n_estimators=250)

    gbc.fit(X_train, y_train)
    y_predict = gbc.predict(X_test)
    y_predict_train = gbc.predict(X_train)
    c_test = classification_report(y_test, y_predict)
    c_train = classification_report(y_train, y_predict_train)
    r = recall_score(y_test, y_predict)
    p = precision_score(y_test, y_predict)

    print("Precision: " + str(p))
    print("Recall:" + str(r))
    print("-*-*-*-Classification Report Test -*-*-*-*-")
    print(c_test)
    print("-*-*-*-Classification Report Train-*-*-*-*-")
    print(c_train)


def train_and_evaluate_vc(X_train, X_test, y_train, y_test):
    vc = VotingClassifier(estimators=[
        ('rfc', RandomForestClassifier(n_estimators=250,
                                       max_depth=50,
                                       min_samples_split=5)),
        ('gbc', GradientBoostingClassifier(n_estimators=250)),
        ('svc', SVC(gamma='scale'))
    ])

    vc.fit(X_train, y_train)
    y_predict = vc.predict(X_test)
    y_predict_train = vc.predict(X_train)
    c_test = classification_report(y_test, y_predict)
    c_train = classification_report(y_train, y_predict_train)
    r = recall_score(y_test, y_predict)
    p = precision_score(y_test, y_predict)

    print("Precision: " + str(p))
    print("Recall:" + str(r))
    print("-*-*-*-Classification Report Test -*-*-*-*-")
    print(c_test)
    print("-*-*-*-Classification Report Train-*-*-*-*-")
    print(c_train)


X, y, all_data = get_train_test_set(dataset)

X_eleminated, y, scores = eleminate_features(X, y, 15)

X_train, X_test, y_train, y_test = train_test_split(X_eleminated, y, test_size=0.2)

params = parameter_selection(X_train, y_train)
print("Found parameters: " + str(params))

train_and_evaluate_rfc(X_train, X_test, y_train, y_test, params)
# train_and_evaluate_rfc(X_train, X_test, y_train, y_test, None)
