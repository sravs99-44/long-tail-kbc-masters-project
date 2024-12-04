import json
import re


mapping = {'residence': ['P551'], 'educated at': ['P69'], 'employer': ['P108'], 'place of birth': ['P19'],
           'place of death': ['P20'], 'founded by': ['P112'],
           'performer': ['P175'], 'composer': ['P86']}

def normalize_text(text):
    """Convert text to lowercase and remove punctuation for comparison."""
    return re.sub(r'[^\w\s]', '', text.lower())

def load_extracted_facts(file_path):
    """Load extracted facts from the text file."""
    extracted_facts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue
            name, fact, relation, confidence, qa_score, ed_score, sentence = parts
            extracted_facts.append({
                "name": normalize_text(name),
                "fact": normalize_text(fact),
                "relation": relation,
                "confidence": float(confidence),
                "qa_score": float(qa_score),
                "ed_score": float(ed_score)
            })
    return extracted_facts

def load_gold_facts(file_path):
    """Load ground truth facts from a JSON Lines file."""
    gold_facts = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            name = normalize_text(obj["name"])  # Normalize name in gold facts
            qid = obj["qid"]
            facts = obj["fact"]
            for rid, fact in facts.items():
                tail_qids = fact["tail_qid"]
                tail_names = {normalize_text(name) for sublist in fact["tail_name"] for name in sublist}  # Normalize tail names
                if name not in gold_facts:
                    gold_facts[name] = []
                gold_facts[name].append({
                    "relation": rid,
                    "tail_qids": set(tail_qids),
                    "tail_names": tail_names
                })
    return gold_facts

def evaluate(extracted_facts, gold_facts):
    """Calculate precision, recall, and F1-score based on extracted and ground truth facts."""
    tp, fp, fn = 0, 0, 0
    index = 0

    for fact in extracted_facts:
        #print("---------------------------------------------------------------------")
        #print("processing fact:", index)
        
        index += 1
        #print(fact)
        name = fact["name"]
        relation = fact["relation"]
        extracted_tail_name = fact["fact"]
        confidence_score = fact["confidence"]
        average_score = (fact["qa_score"] + fact["ed_score"]) / 2
        final_score = max(confidence_score, average_score)  # Take the maximum between confidence and average score
        final_score = confidence_score
        
        # If the final score is below the threshold, count it as a false negative and skip precision tasks
        if confidence_score <= 0.4:
            fn += 1
            #print("It is FALSE NEGATIVE")
            continue 
        
        # Check if this name is in gold facts
        if name in gold_facts:
            matched = False
            for gold_fact in gold_facts[name]:
                #print("---Ground Truth Fact of it---------")
                #print(gold_fact)
                #(gold_fact["relation"])
                #print(relation)

                if extracted_tail_name in gold_fact["tail_qids"]:
                    tp += 1
                    matched = True
                    #print("It is TRUE POSITIVE")
                    break
                    
                for gold_fact in gold_fact["tail_names"]:
                    if extracted_tail_name in gold_fact:
                        tp += 1
                        matched = True
                        #print("It is TRUE POSITIVE")
                        break
  
            if not matched:
                #print("It is FALSE POSITIVE")
                fp += 1
        else:
            fp += 1

    
    #print("<<<<<<<<<<<<<<<tp,fp,fn values>>>>>>>>>>>>>>>>>>")
    #print(tp,fn,fp)

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

# Load the data
#extracted_facts = load_extracted_facts('./data//MALT/extracted_facts.txt')
gold_facts = load_gold_facts('./data/MALT/gold_wikidata.json')

extracted_facts_zeroshot = load_extracted_facts('./data/MALT/extracted_facts_person_chunk_1000_zs.txt')

# Evaluate and print results
#print("The Evaluation Metrics without zero-shot learning:")
#precision, recall, f1_score = evaluate(extracted_facts, gold_facts)
#print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

print("The evaluation metrics with zero-shot learning:")

precision, recall, f1_score = evaluate(extracted_facts_zeroshot, gold_facts)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")