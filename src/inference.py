import argparse
import json, csv
from itertools import chain
from multiprocessing import context
import datasets
from datasets import load_dataset
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from utils_qa_infer import postprocess_qa_predictions

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorForMultipleChoice,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

def parse_args():
    parser = argparse.ArgumentParser(description="Do inference.")
    parser.add_argument(
        "--para_model_dir",
        type=str,
        help="Directory of paragraph selection model.",
    )
    parser.add_argument(
        "--span_model_dir",
        type=str,
        help="Directory of span selection model.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Where to store the final result.",
    )
    parser.add_argument(
        "--context_file",
        type=str,
        help="Directory of the paragraphs.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Directory of the test file.",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if args.context_file is not None:
        with open(args.context_file, 'r', encoding='utf-8') as jsFile:
            context = json.load(jsFile)
            
    # Load data
    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
    
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator_log_kwargs = {}

    # accelerator_log_kwargs["log_with"] = args.report_to
    # accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=2, **accelerator_log_kwargs)

    para_tokenizer = AutoTokenizer.from_pretrained(args.para_model_dir)
    
    para_config = AutoConfig.from_pretrained(args.para_model_dir, trust_remote_code=True)
    
    para_model = AutoModelForMultipleChoice.from_pretrained(
        args.para_model_dir,
        from_tf=bool(".ckpt" in args.para_model_dir),
        config=para_config,
        trust_remote_code=True
    )
    
    para_model.to(device)
    para_model.eval()
    
    para_data_collator = DataCollatorForMultipleChoice(para_tokenizer, pad_to_multiple_of=None)
    # para_data_collator = DataCollatorForMultipleChoice(
    #     para_tokenizer, pad_to_multiple_of=8, return_tensors="pt"
    # )
    
    para_raw_datasets = load_dataset(extension, data_files=data_files)
    
    # para preparation
    para_context_name = "question"
    
    # def get_answer_sentence(examples, i):
    #     if(examples["paragraphs"][i][0] == examples[label_column_name][i]):
    #         return 0
    #     if(examples["paragraphs"][i][1] == examples[label_column_name][i]):
    #         return 1
    #     if(examples["paragraphs"][i][2] == examples[label_column_name][i]):
    #         return 2
    #     if(examples["paragraphs"][i][3] == examples[label_column_name][i]):
    #         return 3
    #     return -1
    
    def para_preprocess_function(examples):
        first_sentences = [[acontext] * 4 for acontext in examples[para_context_name]]
        second_sentences = [
            [context[paragraph[j]] for j in range(0, 4)] for i, paragraph in enumerate(examples["paragraphs"])
        ]
        labels = [0 for i in range(len(examples[para_context_name]))]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = para_tokenizer(
            first_sentences,
            second_sentences,
            max_length=512,
            padding=False,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    para_processed_datasets = para_raw_datasets.map(
        para_preprocess_function, batched=True, remove_columns=para_raw_datasets["test"].column_names
    )
    
    para_dataloader = DataLoader(
        para_processed_datasets["test"], shuffle=False, collate_fn=para_data_collator, batch_size=8
    )
    
    # para inference
    para_predictions = []
    
    for batch in tqdm(para_dataloader):
        for k in batch.keys():
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = para_model(**batch)
        para_predictions.append(outputs.logits.argmax(dim=-1))
    para_predictions = [tensor.cpu().tolist() for tensor in para_predictions]
    flat_para_predictions = [item for sublist in para_predictions for item in sublist]
    # print(flat_para_predictions)
    
    with open(args.test_file, 'r', encoding='utf-8') as jsFile:
        result_json = json.load(jsFile)
        
    for i in range(len(flat_para_predictions)):
        result_json[i]["relevant"] = result_json[i]["paragraphs"][flat_para_predictions[i]]
        
    with open("quicksave.json", 'w', encoding='utf-8') as jsFile:
        json.dump(result_json, jsFile, ensure_ascii=False)
    
    print("Para finished.")
    
    # span model
    
    data_files = {}
    if args.test_file is not None:
        data_files["test"] = "quicksave.json"
        extension = data_files["test"].split(".")[-1]
    
    span_tokenizer = AutoTokenizer.from_pretrained(args.span_model_dir)
    
    span_config = AutoConfig.from_pretrained(args.span_model_dir, trust_remote_code=True)
    
    span_model = AutoModelForQuestionAnswering.from_pretrained(
        args.span_model_dir,
        from_tf=bool(".ckpt" in args.span_model_dir),
        config=span_config,
        use_safetensors=True,
        trust_remote_code=True
    )
    
    span_model.to(device)
    
    span_model.eval()
    
    span_data_collator = DataCollatorWithPadding(span_tokenizer, pad_to_multiple_of=None)
    
    span_raw_datasets = load_dataset(extension, data_files=data_files)
        
    # span preparation
    span_question_column_name = "question"
    span_context_column_name = "relevant"
    pad_on_right = span_tokenizer.padding_side == "right"
    max_seq_length = min(512, span_tokenizer.model_max_length)
    
    def span_prepare_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[span_question_column_name] = [q.lstrip() for q in examples[span_question_column_name]]
        examples[span_context_column_name] = [context[d] for d in examples[span_context_column_name]]
        # examples[span_context_column_name] = [context[example[0]] for i, example in enumerate(examples["paragraphs"])]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = span_tokenizer(
            examples[span_question_column_name if pad_on_right else span_context_column_name],
            examples[span_context_column_name if pad_on_right else span_question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]          

        return tokenized_examples
    
    predict_examples = span_raw_datasets["test"]
    column_names = predict_examples.column_names
    with accelerator.main_process_first():
        predict_dataset = predict_examples.map(
            span_prepare_features,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on prediction dataset",
        )
    
    predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
    span_dataloader = DataLoader(
        predict_dataset_for_model, collate_fn=span_data_collator, batch_size=8
    )
    
    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            flat_para_predictions=flat_para_predictions,
            paragraphs=context,
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False,
            n_best_size=20,
            max_answer_length=30,
            null_score_diff_threshold=0.0,
            output_dir=None,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        return formatted_predictions
    
    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat
    
    all_start_logits = []
    all_end_logits = []
    
    for batch in tqdm(span_dataloader):
        for k in batch.keys():
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = span_model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            # start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
            # end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits
    
    outputs_numpy = (start_logits_concat, end_logits_concat)
    span_prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)

    print("Inference finished.")
    
    with open(args.output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer'])
        for item in span_prediction:
            writer.writerow([item['id'], item['prediction_text']])
        
if __name__ == "__main__":
    main()