# This code is modified from nncp https://bellard.org/nncp/
# Copyright (c) 2024 text-compressor


# coding: utf-8
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import argparse
import numpy as np
import torch
from arithmetic_coder import ArithmeticEncoder
# from datasets import load_dataset
# from itertools import chain
from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder, BitInputStream, BitOutputStream
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='unittest.txt',
                    help='location of the data corpus')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--seqlen', type=int, default=1024,
                    help='the length of the sequence to train on')
parser.add_argument('--output_file', type=str, default='unittest.bin',
                    help='output file for the compressed data')
parser.add_argument('--decompress', action='store_true',
                    help='decompress input to output instead of compressing')

N = 500_000

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

def get_ppl(model, loader): 
    nlls = []
    for data in tqdm(loader):
        input_ids, labels = torch.cat(data['input_ids']).cuda(), torch.cat(data['labels']).cuda() 
        with torch.no_grad():
            val_outputs = model(input_ids, labels=labels)
        neg_log_likelihood = val_outputs.loss

        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


def main():
    args = parser.parse_args()
    set_seed(args.seed)
    # model_name = 'unsloth/Llama-3.2-1B-Instruct'
    model_name = 'openai-community/gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # input_sample = "I love machine learning so much!"
    # input_sample = tokenizer.bos_token + input_sample + tokenizer.eos_token
    # input_ids = tokenizer(input_sample, return_tensors="pt").input_ids

    # attn_implementation='flash_attention_2'
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.double).cuda()
    model.eval()
    
    if not args.decompress:
    ################## Encode ##################
        with open(args.input_file, "r") as lines:
            inputs = ''.join(lines.readlines())
        inputs = tokenizer.bos_token + inputs
        input_ids = tokenizer(inputs, return_tensors="pt").input_ids
        out_file = open(args.output_file, 'wb')
        bit_output = BitOutputStream(out_file)
        arith_enc = ArithmeticEncoder(32, bit_output)
        out_file.write(len(input_ids[0]).to_bytes(4, byteorder='big'))
        input_ids = input_ids.cuda()
        print("original text:", inputs)
        print("original ids:", input_ids)
        labels = input_ids
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits
        logits = torch.round(logits, decimals=2)
        probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.double)
        freq = torch.round(probs * N).int()
        freq = torch.max(freq, freq.new_ones(freq.size()))
        freq = torch.cumsum(freq, -1)
        freq = freq.cpu()
        freq = freq.squeeze(0)
        # freq_tab len(vocab_size)
        for idx in tqdm(range(len(labels[0]) - 1)):
            sym = labels[0][idx + 1].item()
            freq_tab = freq[idx]
            arith_enc.write(freq_tab, sym)

        arith_enc.finish()
        bit_output.close()
        out_file.close()
    else:
    ################## Decode ##################
        in_file = open(args.input_file, "rb")
        original_file_len = int.from_bytes(in_file.read(4), byteorder='big')
        bit_input = BitInputStream(in_file)
        arith_dec = ArithmeticDecoder(32, bit_input)
        input_ids = torch.tensor([[tokenizer.bos_token_id]]).cuda()
        print('Original file length:', original_file_len)
        decoded_tokens = [tokenizer.bos_token_id]
        kvcache = None
        for idx in tqdm(range(0, original_file_len - 1)):
            with torch.no_grad():
                outputs = model(input_ids, past_key_values=kvcache)
            kvcache = outputs.past_key_values
            logits = outputs.logits
            logits = torch.round(logits, decimals=2)
            probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.double)
            freq = torch.round(probs * N).int()
            freq = torch.max(freq, freq.new_ones(freq.size()))
            freq = torch.cumsum(freq, -1)
            freq = freq.cpu()
            freq = freq.squeeze(0)
            freq_tab = freq[-1]
            # freq_tab = tmp[idx]
            # print(idx, torch.max(torch.abs(freq_tab - tmp[idx])))
            # breakpoint()
            sym = arith_dec.read(freq_tab)
            # input_ids = torch.cat([input_ids, torch.tensor([[sym]]).cuda()], dim=-1)
            input_ids = torch.tensor([[sym]]).cuda()
            decoded_tokens.append(sym)
        decoded_text = tokenizer.decode(decoded_tokens)
        print("Decoded text:", decoded_text)
        with open(args.output_file, 'w') as f:
            f.write(decoded_text)


if __name__ == '__main__':
    main()
