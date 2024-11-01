import pickle
from GENRE.genre.trie import Trie
from GENRE.genre.hf_model import GENRE

dict_path = "GENRE/scripts_genre/data/kilt_titles_trie_dict.pkl"
model_path = "GENRE/scripts_genre/models/hf_entity_disambiguation_aidayago"

with open(dict_path, "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

model = GENRE.from_pretrained(model_path).eval()


def debug_trie_get(sent):
    tokens = sent.tolist()  # Convert tensor to list
    trie_output = trie.get(tokens)
    
    if not trie_output:  # If trie.get returns an empty list, use a fallback
        #print(f"No valid tokens found for prefix {tokens}. Returning EOS token.")
        return [model.config.eos_token_id]  # Fallback to EOS token to end generation
    
    return trie_output


# Add a debug print to inspect what's being returned by trie.get
def genre_predict(name, abstract, num_beams=30, top_k=10, templates=None):
    results = dict()
    for template in templates:
        sentences = [
            name[1] + ' ' + abstract + ' ' +
            template.format(a=name[0])
        ]
        predicted = model.sample(
            sentences,
            prefix_allowed_tokens_fn=lambda batch_id, sent: debug_trie_get(sent),
            num_beams=num_beams,
            num_return_sequences=top_k
        )

        predicted = [(pred[0]['text'], round(pred[0]['score'].item(), 5)) for pred in predicted]
        for name, score in predicted:
            if name not in results:
                results[name] = 0
            results[name] += score
    results = sorted(results.items(), key=lambda e:e[1], reverse=True)[:top_k]

    return results





if __name__ == '__main__':
    sentences = """In May 2009, the collaboration of Lhasa de Sela and Patrick Watson was released: 
    the song "Wooden Arms" on his album Wooden Arms. Lhasa de Sela collaborated with [START_ENT] this person [END_ENT]"""
    
    predicted = model.sample(
        [sentences],
        prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
        num_beams=20,
        num_return_sequences=20
    )

    # Print structure of predicted[0] for debugging
    for pred in predicted:
        print(f"Prediction structure: {pred[0]}")  # Debug: print to see what's inside `pred[0]`

    # Handle missing 'logprob' key
    predicted = [(pred[0]['text'], round(pred[0].get('logprob', 0.0), 5)) for pred in predicted]
    
    print(predicted)
