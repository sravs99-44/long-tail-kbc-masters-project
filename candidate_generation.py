from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

def fine_tune_model(model_name="mrm8488/spanbert-finetuned-squadv2"):
    # Load the SQuAD v2 dataset
    squad_dataset = load_dataset("squad_v2")

    # Load model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the dataset
    def preprocess_data(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # Label the answers
        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs.pop("offset_mapping")
        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = inputs["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = inputs.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                token_start_index = 0
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_dataset = squad_dataset.map(preprocess_data, batched=True, remove_columns=squad_dataset["train"].column_names)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=500,
        warmup_steps=500,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    return model, tokenizer


# Candidate Generation Function
def generate(QA_MODEL, name, proof_sentence, top_k=10, tp_number=1, templates=None):
    context = proof_sentence
    pre_results = dict()
    for template in templates[:tp_number]:
        question = template.format(a=name)
        results = QA_MODEL(question=question, context=context, top_k=top_k)
        if len(results) < 1: continue
        for result in results:
            confidence, pred_name, start, end = result['score'], result['answer'], result['start'], result['end']
            scores = [float(r['score']) for r in results if pred_name in r['answer'] or r['answer'] in pred_name]
            scores = sorted(scores)

            score = sum(scores[-10:]) / min(10, len(scores))  # averaging the top scores
            confidence = score
            if pred_name in pre_results: continue
            pre_results[pred_name] = (confidence, start, end)
    return pre_results


if __name__ == '__main__':
    # Fine-tune the QA model with SQuAD v2
    model_name = 'mrm8488/spanbert-finetuned-squadv2'
    model, tokenizer = fine_tune_model(model_name)
    QA_MODEL = pipeline(task='question-answering', model=model, tokenizer=tokenizer, clean_up_tokenization_spaces=False)
    
    # Define candidate generation parameters
    name = 'Tavolevo River'
    pred_name = 'Chile'
    content = 'in Chile.'
    my_templates = ['which country does {a} flow through?']
    
    # Generate candidates
    score = generate(QA_MODEL, name=name, proof_sentence=content, templates=my_templates)
    print(score)
