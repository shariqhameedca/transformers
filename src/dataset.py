import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from utils.common_utils import read_yaml
import os

import warnings

from pathlib import Path

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]

def get_or_build_tokenizer(config, dataset, language):
    tokenizer_tree = config.tokenizer_file.split('/')
    tokenizer_loc = os.path.join(tokenizer_tree[0], tokenizer_tree[1])
    
    if not os.path.exists(tokenizer_loc):
        os.makedirs(tokenizer_loc)

    tokenizer_path = Path(config.tokenizer_file.format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_dataset(config):
    dataset_raw = load_dataset('opus100', f'{config.lang_src}-{config.lang_tgt}', split='train[:30000]')

    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config.lang_tgt)

    train_dataset_size = int(0.9 * len(dataset_raw))
    valid_dataset_size = len(dataset_raw) - train_dataset_size

    train_dataset_raw, valid_dataset_raw = random_split(dataset_raw, [train_dataset_size, valid_dataset_size])

    train_dataset = Bilingual_dataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config.lang_src, config.lang_tgt, config.seq_len)
    valid_dataset = Bilingual_dataset(valid_dataset_raw, tokenizer_src, tokenizer_tgt, config.lang_src, config.lang_tgt, config.seq_len)

    max_len_src = 0
    max_len_tgt = 0

    for item in tqdm(dataset_raw):
        src_ids = tokenizer_src.encode(item['translation'][config.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config.lang_tgt]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    config.seq_len = max(max_len_src, max_len_tgt) + 10

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    return train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tgt


class Bilingual_dataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input, 
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }