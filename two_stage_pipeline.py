import candidate_generation
import corroboration
import math
from transformers import pipeline
import nltk
import template
import argparse
import sys
import torch
import evaluate
import utils
import json
#nltk.download('punkt')

zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
malt_dataset_path = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/malt_eval.txt"
hold_out_dataset_path = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/malt_hold_out.txt"
qa_model = "mrm8488/spanbert-finetuned-squadv2"
wikipedia_dataset = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/wikipedia.json"
wikipedia_subset_dataset = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/wikipedia_subset.json"
person_wikidata = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/person.json"


song_wikidata = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/song.json"
business_wikidata = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/business.json"
gold_path = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/gold_wikidata.json"
output_path = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/extracted_facts.txt"
score_path = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/score.txt" 
score_path_zs = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/score_zs.txt"
output_path_zs = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/extracted_facts_zeroshot.txt"
output_path_person = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/extracted_facts_person.txt"
output_path_person_zs = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/extracted_facts_person_zs.txt"

output_path_song = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/extracted_facts_song.txt"
output_path_song_zs = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/extracted_facts_song_zs.txt"

output_path_business = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/extracted_facts_business.txt"
output_path_business_zs = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/extracted_facts_business_zs.txt"


top_k = 10 
max_len = 1024 
min_can_name_len = 3 
min_sen_len = 30 



"""try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)"""


def load_name():
    train_names = [line.strip().split('\t')[0] for line in
                   open(malt_dataset, encoding='utf8')]
    test_names = [line.strip().split('\t')[0] for line in
                  open(hold_out_dataset, encoding='utf8')]
    return set(train_names + test_names)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
corroboration.model.to(device)

tokenizer = qa_model
model_name = qa_model


if 'cpu' in device:
    device = -1  # Set device to -1 for CPU
else:
    device = int(device.split(':')[-1]) 

model_for_candidate = pipeline(task='question-answering', model=model_name, tokenizer=tokenizer,
                    device=device, clean_up_tokenization_spaces=False)



def run_on_malt(filepath,type,output_path):
    f = open(output_path, 'w', encoding='utf8')
    print("inside run on malt")
    with open("./MALT/business.json", "r") as file:
        data = json.load(file)
        wiki_pages = data
    
    
    for relation_type in template.candidate_templates.keys():
        file_type = template.file_names[relation_type]
        cnt = 0

        for record in wiki_pages:
            #wp_page = record[name]
            e_type = record['type']
            name = record['name']

        #for index, name in enumerate(wiki_pages):
            #print(index,name)
            #break
            #wp_page = wiki_pages[name]
            #e_type = wp_page['type']
            if e_type != file_type: continue
            cnt += 1
            page_content = record['wikipage']
            print('processing {a} lines'.format(a=cnt))
            print('process name = {a} ......'.format(a=record["name"]))
            predict_result = set()
           
            page_contents = nltk.sent_tokenize(page_content)
            
            for page_content in page_contents:
                
                if len(page_content) < min_sen_len: continue
                candidates = candidate_generation.generate(model_for_candidate, name, page_content, top_k=top_k,
                                                  templates=template.candidate_templates[relation_type])
                

                for can_name in candidates:
                    if len(can_name) < min_can_name_len: continue
                    if utils.filter(can_name): continue
                    qa_score, start, end = candidates[can_name]
                    sentence = page_content
                    corroborated_results = corroboration.genre_predict((name, can_name), sentence[:max_len], top_k=top_k, num_beams=top_k,
                                                             templates=template.corroboration_templates[relation_type])
                   
                    corroborated_results = dict([(n, math.exp(v)) for n, v in corroborated_results])
                    
                    for sr in corroborated_results:
                        clean_sr = utils.clean_genre(sr)
                        if can_name != clean_sr: continue
                        if clean_sr in predict_result:continue
                        predict_result.add(clean_sr)
                        ed_score = corroborated_results[sr]
                        avg_score = 0.5 * (qa_score + ed_score)
                        w_l = name + '\t' + clean_sr + '\t' + relation_type + '\t' + str(
                            round(avg_score, 8)) + '\t' + str(
                            round(qa_score, 8)) + '\t' + str(
                            round(ed_score, 8)) + '\t' + sentence + '\n'
                        f.write(w_l)
                        f.flush()

def run_on_malt_zs(filepath, entity_type, output_path):
    f = open(output_path, 'w', encoding='utf8')

    # Load data from JSON file
    with open(filepath, "r") as file:
        wiki_pages = json.load(file)
    
    for relation_type, templates in template.candidate_templates.items():
        file_type = template.file_names[relation_type]
        cnt = 0
        for record in wiki_pages:
            e_type = record['type']
            name = record['name']
            # Skip if entity type does not match
            if e_type != file_type:
                continue
            
            cnt += 1
            page_content = record['wikipage']
            print(f'Processing {cnt} lines for entity {name}...')

            predict_result = set()
            page_contents = nltk.sent_tokenize(page_content)

            for sentence in page_contents:
                # Skip short sentences
                if len(sentence) < min_sen_len:
                    continue

                # Format the candidate labels with the name
                candidate_labels = [t.format(a=name) for t in templates]

                # Generate candidates
                candidates = candidate_generation.generate(model_for_candidate, name, sentence, top_k=top_k, templates=templates)

                

                for can_name in candidates:
                    if len(can_name) < min_can_name_len or utils.filter(can_name):
                        continue  # Skip candidates based on length and filtering
                    # Zero-shot classification for corroboration
                    classification = zero_shot_classifier(
                        sequences=sentence,
                        candidate_labels=candidate_labels,
                        hypothesis_template="{} is related to " + can_name
                    )
                    zero_shot_score = max(classification["scores"])  # Use the highest confidence score

                    qa_score, start, end = candidates[can_name]

                    # Corroborate with genre prediction
                    corroborated_results = corroboration.genre_predict(
                        (name, can_name), sentence[:max_len], top_k=top_k, num_beams=top_k,
                        templates=template.corroboration_templates[relation_type]
                    )
                    corroborated_results = {n: math.exp(v) for n, v in corroborated_results}

                    for sr in corroborated_results:
                        clean_sr = utils.clean_genre(sr)
                        if can_name != clean_sr or clean_sr in predict_result:
                            continue  # Skip if candidate already added

                        predict_result.add(clean_sr)
                        ed_score = corroborated_results[sr]

                        # Combine QA score, corroboration score, and zero-shot score
                        combined_score_zs = 0.4 * qa_score + 0.3 * ed_score + 0.3 * zero_shot_score
                        w_l = f"{name}\t{clean_sr}\t{relation_type}\t{round(combined_score_zs, 8)}\t" \
                              f"{round(qa_score, 8)}\t{round(ed_score, 8)}\t{sentence}\n"

                        # Write output
                        f.write(w_l)
                        f.flush()

    f.close()


def run_example_zero_shot(doc,name):
    
    # Generate candidate names using zero-shot classification to improve relevance
    candidates = candidate_generation.generate(model_for_candidate, name, doc, top_k=top_k,
                                               templates=['the person collaborated with which person?'])
    
    for can_name in candidates:
        if len(can_name) < min_can_name_len: continue
        if utils.filter(can_name): continue
        qa_score, start, end = candidates[can_name]
        sentence = doc
        # Apply zero-shot to enhance corroboration with refined prompt
        corroborated_results = corroboration.genre_predict((name, can_name), sentence[:max_len], top_k=top_k,
                                                           num_beams=top_k,
                                                           templates=['the person {a} collaborated with [START_ENT] this person [END_ENT].'])
        
        corroborated_results = dict([(n, math.exp(v)) for n, v in corroborated_results])
        #print(f"The corroboration results for the candidate {can_name} are:")
        classification = zero_shot_classifier(
            sequences=doc,
            candidate_labels=["collaborator","friend","founder","parent"],
            multi_label=True,
            hypothesis = f'what is the relation between {name} with {can_name}?'
        )
        zero_shot_score = max(classification["scores"])  # Get the highest score for the best label match
        
        for sr in corroborated_results:
            clean_sr = utils.clean_genre(sr)
            if can_name != clean_sr: continue

            # Adjusted average score with zero-shot confidence
            ed_score = corroborated_results[sr]
            
            avg_score = 0.5 * qa_score + 0.25 * ed_score + 0.25 * zero_shot_score
            print('( Lhasa de Sela, collaborator, ' + sr + ', ' + str(avg_score) +' )')

def run_example(doc,name):
   
    candidates = candidate_generation.generate(model_for_candidate, name, doc, top_k=top_k,
                                               templates=['the person {a} colloborated with which person?'])

    #print(type(candidates))
    for can_name in candidates:
        if len(can_name) < min_can_name_len: continue
        #print("candidate: ", can_name)
        #print(utils.filter(can_name))
        if utils.filter(can_name): continue
        
        qa_score, start, end = candidates[can_name]
        #print(qa_score)
        sentence = doc
        corroborated_results = corroboration.genre_predict((name, can_name), sentence[:max_len], top_k=top_k,
                                                           num_beams=top_k,
                                                           templates=['the person {a} colloborated with [START_ENT] this person [END_ENT].'])
        #print("The corroborated results are: ")
        #print(corroborated_results)
        corroborated_results = dict([(n, math.exp(v)) for n, v in corroborated_results])
       
        for sr in corroborated_results:
            clean_sr = utils.clean_genre(sr)
            if can_name != clean_sr: continue
            #print(clean_sr)
            ed_score = corroborated_results[sr]
            avg_score = 0.5 * (qa_score + ed_score)
            print('( Lhasa de Sela, collaborator, ' + sr + ', ' + str(avg_score) +' )')


if __name__ == '__main__':
    doc = """
        Lhasa de Sela said that the song was about inner happiness and
        "feeling my feet in the earth, having a place in the world, of things
        taking care of themselves.“ In May 2009, her collaboration
        with Patrick Watson was released. In 2010 she has played the concert with william James.
    """
    name = 'Lhasa de Sela'
    #print("------------Results without zero-shot-------------")
    #run_example(doc,name)
    #print("-------------Results with zero-shot -------------")
    #run_example_zero_shot(doc,name)
   
    run_on_malt_zs(business_wikidata,"business",output_path_business_zs)
    #evaluate.run_eval([output_path_zs], score_path_zs, malt_dataset_path, hold_out_dataset_path, gold_path)
    #run_on_malt(business_wikidata,"business",output_path_business)
    #evaluate.run_eval([output_path], score_path, malt_dataset_path, hold_out_dataset_path, gold_path)