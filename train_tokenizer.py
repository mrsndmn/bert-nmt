
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from wmt_datamodule import NewsCommentaryTranslationDataset

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='datasets dataset name', type=str, required=True)
    parser.add_argument('--languages', help='dataset languages to tokenize', type=str, required=True)
    parser.add_argument('--tokenizer-out', help='tokenizer output file', type=str, required=True)
    parser.add_argument('--special-tokens', type=str, default="[UNK],[SEP],[PAD],[MASK],[ECHO],[TRANSLATE]")
    args = parser.parse_args()

    # translation_dataset = load_dataset(args.dataset, args.languages)
    # translation_datase    t.set_format(columns='translation')
    translation_dataset = NewsCommentaryTranslationDataset()


    tokenizer_file = args.tokenizer_out
    special_tokens = args.special_tokens.split(",")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=special_tokens)

    all_translation_sentences = map(lambda x: [ x['translation'][lang] for lang in x['translation'].keys() ], translation_dataset)

    tokenizer.train_from_iterator( all_translation_sentences, trainer=trainer )


    tokenizer.save(tokenizer_file)