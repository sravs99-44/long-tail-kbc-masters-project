import json

# Define the types and initialize counters and lists to store records
types = ["person", "song", "business"]
records = {type_name: [] for type_name in types}
counts = {type_name: 0 for type_name in types}
max_records = 10

# Open the JSON file for reading
with open("wikipedia.json", "r") as file:
    for line in file:
        # Parse each JSON record in the line
        json_record = json.loads(line.strip())
        
        # Check the record type and add it to the appropriate list
        record_type = json_record.get("type")  # Adjust if the type key is different
        if record_type in types and counts[record_type] < max_records:
            records[record_type].append(json_record)
            counts[record_type] += 1
        
        # Stop if we've collected 10 records of each type
        if all(count >= max_records for count in counts.values()):
            break

# Write each type's records to its own file
for record_type, record_list in records.items():
    with open(f"{record_type}.json", "w") as output_file:
        json.dump(record_list, output_file, indent=4)

print("10 records of each type have been written to separate files.")
