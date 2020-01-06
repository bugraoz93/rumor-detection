from PhemeDataset import PhemeDatasetES

dataset = PhemeDatasetES(hosts="http://localhost:9200", index_name="pheme_tweet_data")

events = ["charliehebdo", "germanwings-crash", "sydneysiege", "ottawashooting", "ferguson"]
rumor = dict()
non_rumor = dict()
for event in events:
    rumor_avg_timespan = 0
    rumor_count = 0
    non_rumor_avg_timespan = 0
    non_rumor_count = 0
    a = dataset.read_combined_features_from_file(event)
    for t in a:
        if t['rumor'] == 0:
            non_rumor_avg_timespan += t['time_span']
            non_rumor_count += 1
        else:
            rumor_avg_timespan += t['time_span']
            rumor_count += 1

    print("EVENT: " + str(event))
    print("AVG TIMESPAN RUMOR: " + str(rumor_avg_timespan / rumor_count))
    print("AVG TIMESPAN NON-RUMOR: " + str(non_rumor_avg_timespan / non_rumor_count))



