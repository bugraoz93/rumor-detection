from PhemeDataset import PhemeDatasetES

dataset = PhemeDatasetES(hosts="http://localhost:9200", index_name="twitter")

dataset.write_combined_features_to_file()
events = ["charliehebdo", "germanwings-crash", "sydneysiege", "ottawashooting", "ferguson"]
rumor = dict()
non_rumor = dict()
for event in events:
    rumor_ct = 0
    non_rumor_ct = 0
    a = dataset.read_combined_features_from_file(event)
    for i in a:
        if i["rumor"]:
            rumor_ct += 1
        else:
            non_rumor_ct += 1
    rumor[event] = rumor_ct
    non_rumor[event] = non_rumor_ct
print(rumor)
print(non_rumor)