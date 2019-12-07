from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from PhemeDataset import PhemeDatasetES

dataset = PhemeDatasetES(hosts="0.0.0.0:9200", index_name="pheme_tweet_data")

representations = dataset.get_source_tweet_representations("charliehebdo")
representations.extend(dataset.get_source_tweet_representations("germanwings-crash"))
representations.extend(dataset.get_source_tweet_representations("sydneysiege"))
representations.extend(dataset.get_source_tweet_representations("ottawashooting"))
test_set = dataset.get_source_tweet_representations("ferguson")
X = []
y = []
for representation in representations:
    x = [representation['time_span'],
         representation['early_reaction_count'],
         representation['mid_reaction_count'],
         representation['all_reaction_count'],
         representation['reaction_speed'],
         representation['reaction_mention_count'],
         representation['reaction_retweet_count'],
         representation['is_sensitive'],
         representation['user_follower_count'],
         representation['is_verified'],
         representation['user_event_time_diff']
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
         representation['reaction_speed'],
         representation['reaction_mention_count'],
         representation['reaction_retweet_count'],
         representation['is_sensitive'],
         representation['user_follower_count'],
         representation['is_verified'],
         representation['user_event_time_diff']
    ]
    X_test.append(x.copy())
    y_test.append(representation['isRumor'])


#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train = X
y_train = y
def classify_with_rfc(X, y):
    cls = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    cls.fit(X_train, y_train)
    accuracy = cls.score(X_test, y_test)
    print("Accuracy: " + str(accuracy))
    return accuracy, cls


def classify_with_linear_regression(X, y):
    lr = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr.fit(X_train, y_train)
    accuracy = lr.score(X_test, y_test)
    print("Accuracy: " + str(accuracy))
    return accuracy, lr


def classify_with_svm(X, y):
    svc = SVC()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    svc.fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    print("Accuracy: " + str(accuracy))
    return accuracy, svc


def evaluate_model(model, X_test, y_test):
    predicted_y = model.predict(X_test)
    report = classification_report(y_test, predicted_y)
    print(report)


_, model = classify_with_rfc(X_train, y_train)
evaluate_model(model, X_test, y_test)
