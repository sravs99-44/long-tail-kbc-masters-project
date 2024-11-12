import json
import random

gold_names = []
def get_gold_info():
    with open('./MALT/gold_wikidata.json', 'r') as file:
        for line in file:
            json_line = json.dumps(line)




# Read each line as a separate JSON object
data = ""
types = ["person","business","song"]
with open('./MALT/wikipedia.json', 'r') as file:
    for type in types:
        cnt = 0
        for line in file:
            
            json_line = json.loads(line)
            if json_line["type"] == type:
                cnt += 1
                
        
        data.append(json.loads(line.strip()))



# Optionally, save the sample back to a JSON file
"""with open('./MALT/wikipedia_subset.json', 'w') as file:
    for record in sampled_data:
        file.write(json.dumps(record) + '\n')"""