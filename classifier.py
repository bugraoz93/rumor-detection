import copy
import this
import eli5
import pydot
from eli5.sklearn import PermutationImportance
from keras import Model, Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
import shap
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import concatenate
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy

from PhemeDataset import PhemeDatasetES

dataset = PhemeDatasetES(hosts="localhost:9200", index_name="twitter")
charliehebdo = dataset.get_source_tweet_representations("charliehebdo")
germanwings_crash = dataset.get_source_tweet_representations("germanwings-crash")
sydneysiege = dataset.get_source_tweet_representations("sydneysiege")
ottawashooting = dataset.get_source_tweet_representations("ottawashooting")
ferguson = dataset.get_source_tweet_representations("ferguson")
data_sets = {"charliehebdo": charliehebdo, "germanwings-crash": germanwings_crash, "sydneysiege": sydneysiege,
             "ottawashooting": ottawashooting, "ferguson": ferguson}


def create_representation_for_all():
    representations = list()
    for events in data_sets.keys():
        representations.extend(data_sets[events])
        X = []
        X_user = []
        X_tweet = []
        X_propagation = []
        y = []
        for representation in representations:
            x = [
                # User
                representation['is_geo_enabled'],  # 5
                representation['has_description'],  # 6
                representation['description_word_count'],  # 7
                representation['role_score'],  # 8
                representation['user_follower_count'],  # 9
                representation['is_verified'],  # 10
                representation['favorites_count'],  # 11
                # Tweet
                representation['has_question_mark'],  # 12
                representation['question_mark_count'],  # 13
                representation['has_exclamation_mark'],  # 14
                representation['exclamation_mark_count'],  # 15
                representation['has_dotdotdot_mark'],  # 16
                representation['dotdotdot_mark_count'],  # 17
                # propagation
                representation['time_span'],  # 0
                representation['early_reaction_count'],  # 1
                representation['mid_reaction_count'],  # 2
                representation['all_reaction_count'],  # 3
                representation['is_sensitive']  # 4
            ]

            x_user = [
                representation['is_geo_enabled'],  # 0
                representation['has_description'],  # 1
                representation['description_word_count'],  # 2
                representation['role_score'],  # 3
                representation['user_follower_count'],  # 4
                representation['is_verified'],  # 5
                representation['favorites_count'],  # 6
            ]
            x_tweet = [
                representation['has_question_mark'],  # 0
                representation['question_mark_count'],  # 1
                representation['has_exclamation_mark'],  # 2
                representation['exclamation_mark_count'],  # 3
                representation['has_dotdotdot_mark'],  # 4
                representation['dotdotdot_mark_count']  # 5
            ]
            x_propagation = [
                representation['time_span'],  # 0
                representation['early_reaction_count'],  # 1
                representation['mid_reaction_count'],  # 2
                representation['all_reaction_count'],  # 3
                representation['is_sensitive'],  # 4
            ]

            X_user.append(x_user.copy())
            X_tweet.append(x_tweet.copy())
            X_propagation.append(x_propagation.copy())

            X.append(x.copy())
            y.append(representation['isRumor'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        return X, y, X_test, y_test, X_user, X_tweet, X_propagation


def create_representation_for_train(test_set_name):
    representations = list()
    for data_set_name in data_sets.keys():
        if data_set_name == test_set_name:
            test_set = data_sets[data_set_name]
        else:
            representations.extend(data_sets[data_set_name])
    X = []
    y = []
    for representation in representations:
        x = [representation['time_span'],  # 0
             representation['early_reaction_count'],  # 1
             representation['mid_reaction_count'],  # 2
             representation['all_reaction_count'],  # 3
             representation['is_sensitive'],  # 4
             # User
             representation['is_geo_enabled'],  # 5
             representation['has_description'],  # 6
             representation['description_word_count'],  # 7
             representation['role_score'],  # 8
             representation['user_follower_count'],  # 9
             representation['is_verified'],  # 10
             representation['favorites_count'],  # 11
             # Tweet
             representation['has_question_mark'],  # 12
             representation['question_mark_count'],  # 13
             representation['has_exclamation_mark'],  # 14
             representation['exclamation_mark_count'],  # 15
             representation['has_dotdotdot_mark'],  # 16
             representation['dotdotdot_mark_count']  # 17

             ]
        X.append(x.copy())
        y.append(representation['isRumor'])
    X_test = []
    y_test = []
    for representation in test_set:
        x = [representation['time_span'],
             representation['early_reaction_count'],
             representation['mid_reaction_count'],
             representation['all_reaction_count'],
             representation['is_sensitive'],
             # User
             representation['is_geo_enabled'],
             representation['has_description'],
             representation['description_word_count'],
             representation['role_score'],
             representation['user_follower_count'],
             representation['is_verified'],
             representation['favorites_count'],
             # Tweet
             representation['has_question_mark'],
             representation['question_mark_count'],
             representation['has_exclamation_mark'],
             representation['exclamation_mark_count'],
             representation['has_dotdotdot_mark'],
             representation['dotdotdot_mark_count']

             ]
        X_test.append(x.copy())
        y_test.append(representation['isRumor'])

    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X, y, X_test, y_test


def classify_with_rfc(train_data, train_data_y, test_data, test_data_y):
    cls = RandomForestClassifier()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    cls.fit(train_data, train_data_y)
    perm = PermutationImportance(cls, random_state=1).fit(test_data, test_data_y)
    print(eli5.format_as_text(eli5.explain_weights(perm)))
    accuracy = cls.score(test_data, test_data_y)
    print("Accuracy: " + str(accuracy))
    return accuracy, cls


def classify_with_linear_regression(train_data, train_data_y, test_data, test_data_y):
    lr = LinearRegression()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr.fit(train_data, train_data_y)
    perm = PermutationImportance(lr, random_state=1).fit(test_data, test_data_y)
    print(eli5.format_as_text(eli5.explain_weights(perm)))
    accuracy = lr.score(test_data, test_data_y)
    print("Accuracy: " + str(accuracy))
    return accuracy, lr


def classify_with_svm(train_data, train_data_y, test_data, test_data_y):
    svc = SVC()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    svc.fit(train_data, train_data_y)
    perm = PermutationImportance(svc, random_state=1).fit(test_data, test_data_y)
    print(eli5.format_as_text(eli5.explain_weights(perm)))
    accuracy = svc.score(test_data, test_data_y)
    print("Accuracy: " + str(accuracy))
    return accuracy, svc


def classify_with_dt(train_data, train_data_y, test_data, test_data_y):
    svc = DecisionTreeClassifier(random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    svc.fit(train_data, train_data_y)
    perm = PermutationImportance(svc, random_state=1).fit(test_data, test_data_y)
    print(eli5.format_as_text(eli5.explain_weights(perm)))
    # import graphviz
    #
    # tree_graph = tree.export_graphviz(svc, out_file="tree.png")
    #
    # graphviz.Source(tree_graph).save("out")
    accuracy = svc.score(test_data, test_data_y)
    print("Accuracy: " + str(accuracy))
    return accuracy, svc


precisions_rf = list()
precisions_svm = list()
precisions_dt = list()

recall_rf = list()
recall_svm = list()
recall_dt = list()


def evaluate_model(model, X_test, y_test, type):
    predicted_y = model.predict(X_test)
    report = classification_report(y_test, predicted_y)
    matrix = confusion_matrix(y_test, predicted_y)
    precision, recall, fscore, support = score(y_test, predicted_y, average='macro')
    if type == "rf":
        precisions_rf.append(precision)
        recall_rf.append(recall)
    elif type == "svm":
        precisions_svm.append(precision)
        recall_svm.append(recall)
    else:
        precisions_dt.append(precision)
        recall_dt.append(recall)
    print(report)
    print(matrix)


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back + 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back - 1, :])
    return numpy.array(dataX), numpy.array(dataY)


def classify_with_lstm(X, y, X_test, y_test):
    X = numpy.vstack(X)
    y = numpy.vstack(y)
    X_test = numpy.vstack(X_test)
    y_test = numpy.vstack(y_test)
    print(X.shape)
    print(y.shape)
    model = Sequential()
    model.add(LSTM(30, return_sequences=False, input_shape=X.shape))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X, y, epochs=100, batch_size=1, verbose=2)

    scores = model.evaluate(X_test, y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    return model


def select_k_best(i, X, y, X_test):
    print("For the K-Selected feature value: " + str(i))
    # selector = SelectKBest(chi2, k=i)
    # X_train_new = selector.fit_transform(X.copy(), y.copy()).tolist()
    selected_features = [0, 3, 8, 9, 11]
    print("Selected Feature Indices: " + str(selected_features))
    X_train_new = copy.deepcopy(X)
    for data_num in range(len(X)):
        X_train_new[data_num] = [i for j, i in enumerate(X_train_new[data_num]) if j in selected_features]

    X_test_new = copy.deepcopy(X_test)
    for data_num in range(len(X_test_new)):
        X_test_new[data_num] = [i for j, i in enumerate(X_test_new[data_num]) if j in selected_features]
    return X_train_new, X_test_new


def select_eli5(model, X, y, X_test):
    perm = eli5.sklearn.PermutationImportance(model, cv=5)
    perm.fit(X, y)
    sel = SelectFromModel(perm, threshold=0.05, prefit=True)
    X_train_new = sel.transform(X)
    X_test_new = copy.deepcopy(X_test)
    for data_num in range(len(X_test_new)):
        X_test_new[data_num] = [i for j, i in enumerate(X_test_new[data_num]) if j in perm]
    return X_train_new, X_test_new


def shap_selection(model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])


def classify_with_nn(X, y, X_user, X_tweet, X_propagation, X_test, y_test):
    X = numpy.vstack(X)
    print(X.shape)
    y = numpy.vstack(y)
    print(y.shape)
    X_user = numpy.vstack(X_user)
    print(X_user.shape)
    X_tweet = numpy.vstack(X_tweet)
    print(X_tweet.shape)
    X_propagation = numpy.vstack(X_propagation)
    print(X_propagation.shape)

    X_test = numpy.vstack(X_test)
    y_test = numpy.vstack(y_test)

    input_user = Input(shape=(7, ))
    model_user = Dense(14, )(input_user)
    rumor_user = Dense(7, activation='relu')(model_user)

    input_tweet = Input(shape=(6, ))
    model_tweet = Dense(12, )(input_tweet)
    rumor_tweet = Dense(6, activation='relu')(model_tweet)

    input_propagation = Input(shape=(5, ))
    model_propagation = Dense(10, )(input_propagation)
    rumor_propagation = Dense(5, activation='relu')(model_propagation)

    merged_1 = concatenate([rumor_user, rumor_tweet, rumor_propagation])
    before_rumor = Dense(36, activation='relu')(merged_1)
    rumor = Dense(1, activation='relu')(before_rumor)

    model = Model(inputs=[input_user, input_tweet, input_propagation], outputs=rumor)
    print(model.summary())
    pydot.Dot.create(pydot.Dot())
    plot_model(
        model,
        to_file='model.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )

    model.compile(loss='mean_absolute_error', optimizer='Adadelta', metrics=['accuracy'])
    model.fit([X_user, X_tweet, X_propagation], y, epochs=1000, batch_size=4, verbose=2)

    scores = model.evaluate([X_test, X_test, X_test], y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    return model


X, y, X_test, y_test, X_user, X_tweet, X_propagation = create_representation_for_all()
# X_train_new, X_test_new = select_k_best(5, X, y, X_test)

print("NN")
model = classify_with_nn(X, y, X_user, X_tweet, X_propagation, X_test, y_test)
# evaluate_model(model, X_test, y_test, "svm")
print("Random Forrest: ")
# _, model = classify_with_rfc(X, y, X_test, y_test)


print("SVM: ")
# _, model = classify_with_svm(X, y, X_test, y_test)
# evaluate_model(model, X_test, y_test, "svm")

# print("Decision Tree: ")
# _, model = classify_with_dt(X, y, X_test, y_test)
# evaluate_model(model, X_test, y_test, "dt")

# for events in data_sets.keys():
#     X, y, X_test, y_test = create_representation_for_train(events)
#     # for i in range(3, 17):
#     X_train_new, X_test_new = select_k_best(5, X, y, X_test)
#     print("Test Set Name: " + str(events))
#     print("Random Forrest: ")
#     _, model = classify_with_rfc(X_train_new, y, X_test_new, y_test)
#     evaluate_model(model, X_test_new, y_test, "rf")
#
#     print("SVM: ")
#     _, model = classify_with_svm(X_train_new, y, X_test_new, y_test)
#     evaluate_model(model, X_test_new, y_test, "svm")
#
#     print("Decision Tree: ")
#     _, model = classify_with_dt(X_train_new, y, X_test_new, y_test)
#     evaluate_model(model, X_test_new, y_test, "dt")
#     print("----------------------------------------------------")
#
#     # ave_list = list()
#     # for i in range(3, 17):
#     #     pre = (precisions_dt[i - 3] + precisions_rf[i - 3] + precisions_svm[i - 3]) / 3
#     #     rec = (recall_dt[i - 3] + recall_rf[i - 3] + recall_svm[i - 3]) / 3
#     #     ave_list.append({"k": i, "Ave Precision": pre, "Ave Recall": rec})
#     #
#     # max_sco = -1
#     # max_k = 0
#     # for results in ave_list:
#     #     sco = (results["Ave Precision"] * results["Ave Recall"]) / (results["Ave Precision"] + results["Ave Recall"])
#     #     if max_sco < sco:
#     #         max_sco = sco
#     #         max_k = results["k"]
#     #
#     # print(str(max_k) + "----" + str(max_sco))
#
# plt.ylabel('Precision & Recall for RF')
# plt.xlabel('Events')
# x = range(0, 5)
# plt.xticks(x, data_sets.keys())
# plt.plot(x, precisions_rf, label='Precision', marker='o')
# plt.plot(x, recall_rf, label='Recall', marker='x')
# plt.legend()
# plt.show()
#
# plt.ylabel('Precision & Recall for SVM')
# plt.xlabel('Events')
# x = range(0, 5)
# plt.xticks(x, data_sets.keys())
# plt.plot(x, precisions_svm, label='Precision', marker='o')
# plt.plot(x, recall_svm, label='Recall', marker='x')
# plt.legend()
# plt.show()
#
# plt.ylabel('Precision & Recall for DT')
# plt.xlabel('Events')
# x = range(0, 5)
# plt.xticks(x, data_sets.keys())
# plt.plot(x, precisions_dt, label='Precision', marker='o')
# plt.plot(x, recall_dt, label='Recall', marker='x')
# plt.legend()
# plt.show()
