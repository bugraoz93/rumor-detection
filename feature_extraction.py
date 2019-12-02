
from PhemeDataset import PhemeDatasetES


dataset = PhemeDatasetES("0.0.0.0:9200", "pheme_tweet_data")


charlie_data = dataset.get_source_tweet_representations("charliehebdo")

avg_timespan_rumor = 0
avg_timespan_non_rumor = 0
avg_timespan = 0
avg_early_reaction_rumor = 0
avg_early_reaction_non_rumor = 0
avg_mid_reaction_rumor = 0
avg_mid_reaction_non_rumor = 0
avg_all_reaction_rumor = 0
avg_all_reaction_non_rumor = 0
rumor_counter = 0
non_rumor_counter = 0
for data in charlie_data:
    if data['isRumor']:
        avg_timespan_rumor += data['time_span']
        avg_early_reaction_rumor += data['early_reaction_count']
        avg_mid_reaction_rumor += data['mid_reaction_count']
        avg_all_reaction_rumor += data['all_reaction_count']
        rumor_counter += 1
    else:
        avg_timespan_non_rumor += data['time_span']
        avg_early_reaction_non_rumor += data['early_reaction_count']
        avg_mid_reaction_non_rumor += data['mid_reaction_count']
        avg_all_reaction_non_rumor += data['all_reaction_count']
        non_rumor_counter += 1

    avg_timespan += data['time_span']

avg_timespan_rumor = avg_timespan_rumor / rumor_counter
avg_timespan_non_rumor = avg_timespan_non_rumor / non_rumor_counter
avg_timespan = avg_timespan / (non_rumor_counter + rumor_counter)
avg_early_reaction_rumor = avg_early_reaction_rumor / rumor_counter
avg_mid_reaction_rumor = avg_mid_reaction_rumor / rumor_counter
avg_all_reaction_rumor = avg_all_reaction_rumor / rumor_counter
avg_early_reaction_non_rumor = avg_early_reaction_non_rumor / non_rumor_counter
avg_mid_reaction_non_rumor = avg_mid_reaction_non_rumor / non_rumor_counter
avg_all_reaction_non_rumor = avg_all_reaction_non_rumor / non_rumor_counter



print(charlie_data)
