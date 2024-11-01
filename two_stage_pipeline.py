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

zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


#nltk.download('punkt')

malt_dataset_path = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/malt_subset.txt"
hold_out_dataset_path = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/malt_hold_out.txt"
qa_model = "mrm8488/spanbert-finetuned-squadv2"
wikipedia_dataset = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/wikipedia.json"
gold_path = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/gold_wikidata.json"

output_path = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/extracted_facts.txt"
score_path = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/score.txt" 
output_path_zs = "/Users/sravanimalla/Documents/GitHub/long_tail_kbc/MALT/extracted_facts_zeroshot.txt"

top_k = 10 
max_len = 1024 
min_can_name_len = 3 
min_sen_len = 30 



#hyper-parameters
parser = argparse.ArgumentParser(description='contrastive learning framework for word vector')
#parser.add_argument('-malt_dataset', help='the file path of the MALT evaluation dataset', type=str, default='MALT/malt_eval.txt')
#parser.add_argument('-hold_out_dataset', help='the hold-out dataset', type=str, default='MALT/malt_hold_out.txt')
#parser.add_argument('-qa_model', help='the name of the qa model for candidate generation ', type=str, default='mrm8488/spanbert-finetuned-squadv2')
#parser.add_argument('-output_path', help='the file of extracted facts', type=str, default='extracted_facts.txt')
#parser.add_argument('-score_path', help='the file to store the eval score', type=str, default='score.txt')
#parser.add_argument('-wikipedia_dataset', help='Wikipedia pages ', type=str, default='MALT/wikipedia.json')
#parser.add_argument('-top_k', help='top_k', type=int, default=20)
#parser.add_argument('-max_len', help='the maximum length of an input context sentence', type=int, default=1024)
#parser.add_argument('-min_can_name_len', help='the minimum length of a candidate name', type=int, default=3)
#parser.add_argument('-min_sen_len', help='the minimum length of a sentence', type=int, default=30)
parser.add_argument('-run_example', help='if run the example', type=bool, default=False)


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


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


def run_on_malt():
    f = open(output_path, 'w', encoding='utf8')
    print("inside run on malt")

    wiki_pages = utils.load_wiki_page(path=wikipedia_dataset)

    for relation_type in template.candidate_templates.keys():
        file_type = template.file_names[relation_type]

        cnt = 0
        for index, name in enumerate(wiki_pages):
            print(index,name)

            wp_page = wiki_pages[name]
            e_type = wp_page['type']
            if e_type != file_type: continue
            cnt += 1
            page_content = wp_page['wikipage']
            print('processing {a} lines'.format(a=cnt))
            print('process name = {a} ......'.format(a=name))

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


def run_on_malt_zs():
    f = open(output_path_zs, 'w', encoding='utf8')
    print("inside run on malt")

    wiki_pages = utils.load_wiki_page(path=wikipedia_dataset)

    for relation_type in template.candidate_templates.keys():
        file_type = template.file_names[relation_type]

        cnt = 0
        for index, name in enumerate(wiki_pages):
            print(index, name)

            wp_page = wiki_pages[name]
            e_type = wp_page['type']
            if e_type != file_type: continue
            cnt += 1
            page_content = wp_page['wikipage']
            print('processing {a} lines'.format(a=cnt))
            print('process name = {a} ......'.format(a=name))

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
                    
                    # Use corroboration as before
                    corroborated_results = corroboration.genre_predict((name, can_name), sentence[:max_len], top_k=top_k, num_beams=top_k,
                                                             templates=template.corroboration_templates[relation_type])
                    corroborated_results = dict([(n, math.exp(v)) for n, v in corroborated_results])

                    for sr in corroborated_results:
                        clean_sr = utils.clean_genre(sr)
                        if can_name != clean_sr: continue

                        if clean_sr in predict_result: continue
                        predict_result.add(clean_sr)

                        ed_score = corroborated_results[sr]

                        # Apply zero-shot learning to classify the relationship
                        candidate_labels = ["collaborator", "enemy", "friend", "teammate", "colleague"]
                        zero_shot_result = zero_shot_classifier(f"{name} has a {relation_type} with {clean_sr}.", candidate_labels)
                        zero_shot_score = zero_shot_result['scores'][0]  # Confidence score for the top label
                        predicted_label = zero_shot_result['labels'][0]

                        print(f"Zero-shot label: {predicted_label} with confidence: {zero_shot_score}")
                        
                        # Combine QA score, corroboration score, and zero-shot score
                        combined_score = 0.4 * qa_score + 0.4 * ed_score + 0.2 * zero_shot_score
                        
                        # Write the result to the file
                        w_l = name + '\t' + clean_sr + '\t' + relation_type + '\t' + str(
                            round(combined_score, 8)) + '\t' + str(
                            round(qa_score, 8)) + '\t' + str(
                            round(ed_score, 8)) + '\t' + str(
                            round(zero_shot_score, 8)) + '\t' + sentence + '\n'
                        
                        f.write(w_l)
                        f.flush()

    f.close()

def run_example_zero_shot():
    doc = """
        Lhasa de Sela said that the song was about inner happiness and
        "feeling my feet in the earth, having a place in the world, of things
        taking care of themselves.“ In May 2009, her collaboration
        with Patrick Watson was released. 
    """
    name = 'Lhasa de Sela'
    candidates = candidate_generation.generate(model_for_candidate, name, doc, top_k=top_k,
                                               templates=['the person collaborated with which person?'])

    for candidate in candidates:
        if len(candidate) < min_can_name_len: continue
        if utils.filter(candidate): continue
        qa_score, start, end = candidates[candidate]
        #print(f"Candidate: {candidate}, QA Score: {qa_score}")

        # Apply zero-shot classification to assess candidate confidence
        candidate_labels = ["collaborator", "friend", "artist", "musician"]
        zero_shot_results = zero_shot_classifier(f"{name} collaborated with {candidate}.", candidate_labels)
        
        # Get the score for the top prediction
        zero_shot_score = zero_shot_results['scores'][0]  # Confidence score of the top label
        predicted_label = zero_shot_results['labels'][0]  # Top predicted label
        #print(f"Zero-shot Prediction: {predicted_label} with confidence: {zero_shot_score}")

        # Combine QA score with zero-shot score
        combined_score = (qa_score + zero_shot_score) / 2
        #print(f"Combined Score for {candidate}: {combined_score}")

        # Proceed with corroboration using the new combined score
        sentence = doc
        corroborated_results = corroboration.genre_predict((name, candidate), sentence[:max_len], top_k=top_k,
                                                           num_beams=top_k,
                                                           templates=['the person {a} collaborated with [START_ENT] this person [END_ENT].'])
        corroborated_results = dict([(n, math.exp(v)) for n, v in corroborated_results])

        for sr in corroborated_results:
            clean_sr = utils.clean_genre(sr)
            if candidate != clean_sr: continue

            ed_score = corroborated_results[sr]
            avg_score = (combined_score + ed_score) / 2  # Average with corroboration score
            print(f'( {name}, collaborator, {sr}, {avg_score} )')

def run_example():
    doc = """
        Lhasa de Sela said that the song was about inner happiness and
        "feeling my feet in the earth, having a place in the world, of things
        taking care of themselves.“ In May 2009, her collaboration
        with Patrick Watson was released. 
    """
    name = 'Lhasa de Sela'
    candidates = candidate_generation.generate(model_for_candidate, name, doc, top_k=top_k,
                                               templates=['the person colloborated with which person?'])
    

    
   

    for can_name in candidates:
        #print(can_name)
        if len(can_name) < min_can_name_len: continue
        if utils.filter(can_name): continue
        qa_score, start, end = candidates[can_name]
       
        sentence = doc
       
        
        corroborated_results = corroboration.genre_predict((name, can_name), sentence[:max_len], top_k=top_k,
                                                           num_beams=top_k,
                                                           templates=['the person {a} colloborated with [START_ENT] this person [END_ENT].'])
        corroborated_results = dict([(n, math.exp(v)) for n, v in corroborated_results])

        #print(corroborated_results)
        for sr in corroborated_results:
            clean_sr = utils.clean_genre(sr)
            if can_name != clean_sr: continue

            ed_score = corroborated_results[sr]
            avg_score = 0.5 * (qa_score + ed_score)
            print('( Lhasa de Sela, collaborator, ' + sr + ', ' + str(avg_score) +' )')


if __name__ == '__main__':
    print("Results without zero-shot-------------")
    run_example()
    print("Results with zero-shot -------------")
    run_example_zero_shot()
    #run_on_malt_zs()
    #evaluate.run_eval([output_path_zs], score_path, malt_dataset_path, hold_out_dataset_path, gold_path)