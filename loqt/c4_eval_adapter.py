from datasets import load_dataset, Dataset
import torch
import random
from transformers import AutoTokenizer

def get_c4(tokenizer, nsamples, seed, seqlen):
    print("get c4")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    random.seed(seed)
    print(traindata)
    print(valdata)
    print(f"elem1 valdataset: {valdata[0]}")
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(32):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc_stacked = torch.hstack(valenc)
    print(f"valenc shape: {valenc_stacked.shape}")
        # Create a Hugging Face Dataset from the trainloader
    train_dataset = Dataset.from_dict({
        'text': [x[0] for x in trainloader],
        'labels': [x[1] for x in trainloader]
    })
    print(valenc)
    # De-tokenize valenc and get the text
    # valenc_text = [tokenizer.decode(v.tolist(), skip_special_tokens=True) for q in for v in valenc]
        # De-tokenize valenc and get the text
    sequence_list = []
    for seq in valenc:
        res_string = ""
        for token_id in seq.tolist()[0]:
            res_string += tokenizer.decode(token_id, skip_special_tokens=True)
        sequence_list.append(res_string)
    # valenc_text = [tokenizer.decode(token_id, skip_special_tokens=True) 
    #                for v in valenc for token_id in v.tolist()[0]]
    val_dataset = Dataset.from_dict({'text': sequence_list})
    print(val_dataset)
    print(f"elem1 valdataset_constructed: {val_dataset[0]}")
    return train_dataset, val_dataset

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", model_max_length=10000)
n_samples = 128
seed = 2
seqlen = 128
dataset = get_c4(tokenizer, n_samples, seed, seqlen)