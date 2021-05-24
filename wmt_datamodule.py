
import typing
from argparse import ArgumentParser
import math
import functools
from dataclasses import dataclass

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class EncodingBatchedSubset:
    def __init__(self, eb, ranges):
        self.eb = eb
        self.ranges = ranges

    @property
    def tokens_ids(self):
        return self.eb.tokens_ids[self.ranges[0]:self.ranges[1], :]

    @property
    def attention_masks(self):
        return self.eb.attention_masks[self.ranges[0]:self.ranges[1], :]

    @property
    def special_tokens_masks(self):
        return self.eb.special_tokens_masks[self.ranges[0]:self.ranges[1], :]

class EncodingBatched:
    def __init__(self, batch, device='cpu'):
        self.batch = batch
        self.device = device

    def __len__(self):
        return len(self.batch)

    @functools.cached_property
    def tokens_ids(self):
        token_ids = list( map(lambda x: x.ids, self.batch) )
        ret = torch.LongTensor(token_ids)
        return ret.to(self.device)

    @functools.cached_property
    def attention_masks(self):
        attention_masks = list( map(lambda x: x.attention_mask, self.batch) )
        ret = torch.LongTensor(attention_masks)
        return ret.to(self.device)

    @functools.cached_property
    def special_tokens_masks(self):
        special_tokens_masks = list( map(lambda x: x.special_tokens_mask, self.batch) )
        ret = torch.LongTensor(special_tokens_masks)
        return ret.to(self.device)

    def halve(self) -> typing.Tuple[EncodingBatchedSubset, EncodingBatchedSubset]:
        batch_size = len(self.batch)
        return EncodingBatchedSubset(self, (0, batch_size // 2)), EncodingBatchedSubset(self, (batch_size // 2, batch_size))

    @functools.cached_property
    def src_encoding(self):
        return self.halve()[0]

    @functools.cached_property
    def trg_encoding(self):
        return self.halve()[1]


class WMT20DataModule(pl.LightningDataModule):

    name = "wmt20"

    def __init__(self,
                 batch_size: int = 1,
                 val_batch_size: int = None,
                 dataset=None,
                 languages=None,
                 tokenizer: Tokenizer=None,
                 device='cpu',
                 ):
        super(WMT20DataModule, self).__init__()

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size

        if dataset is None:
            raise ValueError(f"dataset is required for {self}")
        self.dataset = dataset
        if languages is None:
            raise ValueError(f"languages is required for {self}")
        self.languages = languages
        self.tokenizer = tokenizer
        pad_token = "[PAD]"
        self.tokenizer.enable_padding(pad_id=tokenizer.token_to_id(pad_token), pad_token=pad_token)

        translate_postprocessor = TemplateProcessing(
            single="[TRANSLATE] $0 [SEP]",
            special_tokens=[("[TRANSLATE]", tokenizer.token_to_id('[TRANSLATE]')), ("[SEP]", tokenizer.token_to_id('[SEP]'))],
        )

        tokenizer.post_processor = translate_postprocessor

        self.device = device

        return

    def sort_by_len(self, dataset):
        return sorted(dataset, key=lambda x: len(x['translation']['en']))

    def setup(self, stage=None):

        ncd = NewsCommentaryTranslationDataset()

        valid_len = 20000
        self.train, self.valid = torch.utils.data.random_split(ncd, [len(ncd) - valid_len, valid_len], generator=torch.Generator().manual_seed(42))

        # translation_dataset = load_dataset(self.dataset, self.languages)
        # translation_dataset.set_format(columns='translation')

        # self.train = self.sort_by_len(translation_dataset['train'])
        # self.valid = self.sort_by_len(translation_dataset['validation'])
        # self.test  = self.sort_by_len(translation_dataset['test'])

        return

    def collate_fn(self, batch: typing.List) -> EncodingBatched:

        src_lang = 'en'
        trg_lang = 'ru'

        src_lang_sentences = []
        trg_lang_sentences = []
        for item in batch:
            src_lang_sentences.append( item['translation'][src_lang] )
            trg_lang_sentences.append( item['translation'][trg_lang] )

        encoded_batch = self.tokenizer.encode_batch( src_lang_sentences + trg_lang_sentences )

        return EncodingBatched(encoded_batch, device=self.device)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True, num_workers=1)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid, batch_size=self.val_batch_size, collate_fn=self.collate_fn, shuffle=False, num_workers=1)


class NewsCommentaryTranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_path='.data/SRC_news-commentary-v14.en-ru.tsv', trg_path='.data/TRG_news-commentary-v14.en-ru.tsv'):

        norm_len_indexes = []
        with open(src_path, 'r') as f:
            self.src = list(map(lambda x: x.strip(), f.readlines()))

        with open(trg_path, 'r') as f:
            self.trg = list(map(lambda x: x.strip(), f.readlines()))

        for i, x in enumerate(self.src):
            if len(x) < 300:
                norm_len_indexes.append(i)

        print(f"Filtered by len { len(self.src) - len(norm_len_indexes) } sentences. Rest: { len(norm_len_indexes) }")

        self.src = [self.src[i] for i in norm_len_indexes]
        self.trg = [self.trg[i] for i in norm_len_indexes]

        assert len(self.src) == len(self.trg)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return { "translation": { "en": self.trg[i], "ru": self.src[i] } }
