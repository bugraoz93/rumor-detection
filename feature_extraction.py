
from PhemeDataset import PhemeDatasetES


dataset = PhemeDatasetES("localhost:9200", "twitter")


charlie_data = dataset.get_event_time_frames_with_time('charliehebdo', 50)


