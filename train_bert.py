import random
import typing
from argparse import ArgumentParser
import math
from nltk.translate.bleu_score import corpus_bleu
from copy import deepcopy

import transformers.modeling_outputs as modeling_outputs
from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert import BertTokenizer

from tokenizers import Tokenizer

import torch
import torch.nn as nn
import pytorch_lightning as pl


from wmt_datamodule import EncodingBatched, WMT20DataModule

class DoNothing(nn.Module):
    def forward(self, embeddings):
        return embeddings

class BertModelInvertibleEmbeddings(BertModel):
    def __init__(self, *args, **kwargs):
        super(BertModelInvertibleEmbeddings, self).__init__(*args, **kwargs)

        self.encoder.layer[-1].output.dropout.p = 0
        self.encoder.layer[-1].output.LayerNorm = DoNothing()

        self.encoder.layer[0].output.LayerNorm = DoNothing()

        return

    def get_raw_embeddings(self, tokens_ids=None, special_tokens_masks=None):
        layer_norm = self.embeddings.LayerNorm

        self.embeddings.LayerNorm = DoNothing()

        was_dropout_trainig = self.embeddings.training
        eval_emb = self.embeddings.eval()
        tokens_embeddings = eval_emb.forward(
            input_ids=tokens_ids,
            token_type_ids=special_tokens_masks,
        )

        if was_dropout_trainig:
            self.embeddings.train()

        self.embeddings.LayerNorm = layer_norm


        return tokens_embeddings

    def word_embeddings_from_lhs(self, last_hidden_state, position_ids=None, token_type_ids=None, past_key_values_length=0):
        # last_hidden_state [ bs, seq_len, hidden_dim ]

        batch_size = last_hidden_state.size(0)
        seq_length = last_hidden_state.size(1)

        if position_ids is None:
            position_ids = self.embeddings.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros( ( batch_size, seq_length ) , dtype=torch.long, device=self.embeddings.position_ids.device)

        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

        if self.embeddings.position_embedding_type == "absolute":
            position_embeddings = self.embeddings.position_embeddings(position_ids)

        word_embeddings = last_hidden_state.clone() - position_embeddings - token_type_embeddings

        return word_embeddings

    def tokens_from_words_embeddings(self, words_embeddings):
        with torch.no_grad():
            # self.embeddings.word_embeddings.weight ~ [ vocab_size x hidden_dim ]
            # words_embeddings ~ [ batch_size, seq_len, hidden_dim ]

            predicted_token_ids = torch.zeros( (words_embeddings.size(0), words_embeddings.size(1)), dtype=torch.long, device=words_embeddings.device )

            for batch_i in range(words_embeddings.size(0)):
                for seq_len_i in range(words_embeddings.size(1)):
                    word_emb = words_embeddings[batch_i, seq_len_i, :]

                    # todo possible another metrics to count tokens similarity
                    # to get similar sentences it is possible to apply beam search to most similar tokens
                    decode_diff = self.embeddings.word_embeddings.weight - word_emb
                    decode_diff = decode_diff.abs()
                    decode_diff = decode_diff.sum(dim=1)

                    predicted_token_ids[batch_i, seq_len_i] = torch.argmin(decode_diff)

            # [batch_size, seq_len]
            return predicted_token_ids




class BertLightningModule(pl.LightningModule):

    all_hyperparameters_list = [
        "lr",
        "scheduler", "noam_opt_warmup_steps", "noam_step_factor", 'noam_scaler',
        "emb_norm_reg"
    ]

    def __init__(self,
        lr: float = 1., # see also lr scheduler
        noam_opt_warmup_steps: int= 4000,
        scheduler: str="noam",
        scheduler_patience:int=10,
        noam_step_factor: float = 1.,
        noam_scaler: float = 1.,
        emb_norm_reg = 0.001,
        tokenizer: Tokenizer = None,
        **kwargs,
    ):

        super(BertLightningModule, self).__init__()

        devbert_config = BertConfig.from_dict({
            "attention_probs_dropout_prob": 0.05,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.05,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",

            "num_attention_heads": 8,
            "num_hidden_layers": 12,

            "pad_token_id": 0,

            "type_vocab_size": 2, # todo increase type vocab size
            "vocab_size": 30000,
        })

        self.save_hyperparameters( *self.all_hyperparameters_list )

        self.bertmodel = BertForMaskedLM(config=devbert_config)

        self.tokenizer: Tokenizer = tokenizer

        self.logged_target_tokens = False

        return

    def get_non_pad_embeddings(self, hidden_states, attention_masks):

        embeddings = []

        for batch_n in range(attention_masks.size(0)):
            for seq_n in range(attention_masks.size(1)):
                if attention_masks[batch_n, seq_n] == 0:
                    break
                embeddings.append( hidden_states[batch_n, seq_n, :] )

        result = torch.vstack(embeddings)

        assert result.size(0) == attention_masks.sum()
        assert result.size(1) == hidden_states.size(-1)

        return result

    def compute_mlm_loss(self, batch: EncodingBatched, decode_tokens=False):

        mlm_batch = deepcopy(batch)
        mlm_batch.tokens_ids[:, 0] = self.tokenizer.token_to_id('[ECHO]')

        batch_echo = deepcopy(mlm_batch)

        tokens_to_mask = (batch_echo.special_tokens_masks == 0) & ( torch.empty(batch_echo.tokens_ids.size(), device=batch_echo.tokens_ids.device).uniform_() > 0.9 )

        mask_token_id = self.tokenizer.token_to_id('[MASK]')
        batch_echo.tokens_ids[tokens_to_mask] = mask_token_id
        batch_echo.special_tokens_masks[tokens_to_mask] = 1

        mlm_bertout: modeling_outputs.MaskedLMOutput = self.bertmodel.forward(
            input_ids=batch_echo.tokens_ids,
            attention_mask=batch_echo.attention_masks,
            token_type_ids=batch_echo.special_tokens_masks,
            labels=mlm_batch.tokens_ids
        )

        if decode_tokens:
            logits = mlm_bertout.logits.view(-1, self.bertmodel.config.vocab_size)
            _, predicted_tokens = logits.max(dim=1)
            predicted_tokens = predicted_tokens.reshape(mlm_batch.tokens_ids.size())

            real_tokens_mask = (mlm_batch.attention_masks == 1)
            tokens_matched = ((predicted_tokens == mlm_batch.tokens_ids) & real_tokens_mask).sum()

            self.log( "tokens_matched", tokens_matched.item() )
            tokens_matched_accuracy = tokens_matched / real_tokens_mask.sum().item()
            self.log( "tokens_matched_accuracy",  tokens_matched_accuracy.item(), prog_bar=True )

            mask_token_id = self.tokenizer.token_to_id('[MASK]')
            self.log("num_decoded_masks", (predicted_tokens == mask_token_id).sum().item())

            unk_token_id = self.tokenizer.token_to_id('[UNK]')
            self.log("num_decoded_unkns", (predicted_tokens == unk_token_id).sum().item())

            return mlm_bertout, mlm_bertout.loss, predicted_tokens

        return mlm_bertout, mlm_bertout.loss

    def training_step(self, batch: EncodingBatched, batch_idx):

        # sequential!
        # MLM
        # BrokenMLM
        # Translate
        # FixLM

        mlm_bertout, loss = self.compute_mlm_loss(batch)
        self.log( "loss", loss.item())

        opt = self.optimizers()
        self.log("lr", opt.param_groups[0]['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch: EncodingBatched, batch_idx: int):

        mlm_bertout, mlm_loss, decoded_tokens = self.compute_mlm_loss(batch, decode_tokens=True)

        self.log( "valid_loss", mlm_loss.item() )

        return batch.tokens_ids, decoded_tokens

    def validation_epoch_end(self, valid_steps_outs):

        berts_tokens = []
        target_tokens = []
        for batch in valid_steps_outs:
            batch_bert_tokens = self.tokenizer.decode_batch(batch[1].detach().cpu().numpy().tolist(), skip_special_tokens=False)

            berts_tokens.extend(batch_bert_tokens)

        berts_tokens_log = random.sample(berts_tokens, min(len(berts_tokens), 5))
        self.logger.experiment.add_text("bert_outputs", "\n\n\n".join(berts_tokens_log))


        if not self.logged_target_tokens:
            for batch in valid_steps_outs:

                batch_target_tokens  = self.tokenizer.decode_batch(batch[0].detach().cpu().numpy().tolist(), skip_special_tokens=False)
                target_tokens.extend(batch_target_tokens)

            self.logger.experiment.add_text("target_outputs", "\n\n\n".join(target_tokens))
            self.logged_target_tokens = True

        return

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--noam_opt_warmup_steps", type=int)

        parser.add_argument("--lr", type=float, default=1)
        parser.add_argument("--scheduler", default="noam")
        parser.add_argument("--scheduler_patience")
        parser.add_argument("--noam_step_factor", type=float, default=1)
        parser.add_argument("--noam_scaler", type=float, default=1)
        parser.add_argument("--emb_norm_reg", type=float, default=0.001)

        return parser

    def noam_opt(self, current_step: int):
        current_step = self.trainer.global_step * self.hparams.noam_step_factor
        min_inv_sqrt = min(1/math.sqrt(current_step+1), current_step * self.hparams.noam_opt_warmup_steps ** (-1.5))
        current_lr = min_inv_sqrt / math.sqrt(self.bertmodel.config.hidden_size)
        current_lr *= self.hparams.noam_scaler
        return current_lr

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.bertmodel.parameters(), lr=self.hparams.lr)

        if self.hparams.scheduler == "no":
            return opt
        elif self.hparams.scheduler == "noam":
            opt_sched = torch.optim.lr_scheduler.LambdaLR(opt, self.noam_opt)
        elif self.hparams.scheduler == "pletau":
            scheduler_patience = self.hparams.scheduler_patience
            if scheduler_patience is None:
                scheduler_patience = 10
            opt_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=scheduler_patience, min_lr=1e-5, factor=0.5, verbose=True)
        else:
            raise ValueError("unknown scheduler " + self.hparams.scheduler)


        return [opt], [{"scheduler": opt_sched, "interval": "step", "monitor": "loss"}]

class BertIELightningModule(BertLightningModule):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

        self.bertmodel = BertModelInvertibleEmbeddings(devbert_config, add_pooling_layer=False)
        # self.criterion = nn.L1Loss(reduction='sum')
        self.criterion = nn.TripletMarginWithDistanceLoss(reduction='mean', margin=1)
        self.distance = nn.PairwiseDistance()

    def compute_mlm_loss(self, batch: EncodingBatched, decode_tokens=False):

        mlm_batch = deepcopy(batch)
        mlm_batch.tokens_ids[:, 0] = self.tokenizer.token_to_id('[ECHO]')

        batch_echo = deepcopy(mlm_batch)

        tokens_to_mask = (batch_echo.special_tokens_masks == 0) & ( torch.empty(batch_echo.tokens_ids.size(), device=batch_echo.tokens_ids.device).uniform_() > 0.9 )

        mask_token_id = self.tokenizer.token_to_id('[MASK]')
        batch_echo.tokens_ids[tokens_to_mask] = mask_token_id
        batch_echo.special_tokens_masks[tokens_to_mask] = 1

        mlm_bertout: modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions = self.bertmodel.forward(
            input_ids=batch_echo.tokens_ids,
            attention_mask=batch_echo.attention_masks,
            token_type_ids=batch_echo.special_tokens_masks,
        )

        with torch.no_grad():
            mlm_tokens_embeddings = self.bertmodel.get_raw_embeddings(
                tokens_ids=mlm_batch.tokens_ids,
                special_tokens_masks=mlm_batch.special_tokens_masks
            )

        hidden_size = mlm_tokens_embeddings.size(2)

        # ignore pad index
        # pad_tokens_mask = (batch_echo.attention_masks == 0).unsqueeze(-1).repeat(1, 1, hidden_size)
        # mlm_bertout.last_hidden_state[pad_tokens_mask] = mlm_tokens_embeddings[pad_tokens_mask]
        # mlm_tokens_embeddings.masked_fill_( pad_tokens_mask, 0 )

        anchor = self.get_non_pad_embeddings(mlm_bertout.last_hidden_state, batch_echo.attention_masks) # .view( -1, hidden_size )
        positive = self.get_non_pad_embeddings(mlm_tokens_embeddings, batch_echo.attention_masks) # .view( -1, hidden_size )

        negative_tokens = torch.randint_like(
            batch_echo.tokens_ids,
            high=self.bertmodel.config.vocab_size,
            device=batch_echo.tokens_ids.device,
            dtype=batch_echo.tokens_ids.dtype
        )

        with torch.no_grad():
            negative = self.bertmodel.get_raw_embeddings( tokens_ids=negative_tokens )
            negative = self.get_non_pad_embeddings(negative, batch_echo.attention_masks) # .view( -1, hidden_size )


        # active_tokens_cnt = (batch_echo.attention_masks == 1).sum().item()

        # print("anchor", anchor.shape, "positive", positive.shape, "negative", negative.shape)
        tmwd_loss = self.criterion( anchor, positive, negative )
        self.log('tmwd_loss', tmwd_loss.item())

        positive_loss = self.distance(anchor, positive).mean()
        self.log('positive_loss', positive_loss.item())

        mlm_loss = tmwd_loss + positive_loss * 0.5

        # tokens_to_mask_cnt = tokens_to_mask.sum()
        # if tokens_to_mask_cnt > 0:
        #     masks_selector = tokens_to_mask.unsqueeze(-1).repeat(1, 1, mlm_tokens_embeddings.size(2))
        #     masked_lhs_embeddings = mlm_bertout.last_hidden_state[masks_selector]
        #     masked_mlm_tokens_embeddings = mlm_tokens_embeddings[masks_selector]
        #     masks_loss = self.criterion( masked_lhs_embeddings, masked_mlm_tokens_embeddings ) / (tokens_to_mask_cnt * hidden_size)
        #     self.log("masks_loss", masks_loss)
        #     mlm_loss += masks_loss

        if decode_tokens:
            word_embeddings = self.bertmodel.word_embeddings_from_lhs(mlm_bertout.last_hidden_state)
            decoded_tokens = self.bertmodel.tokens_from_words_embeddings(word_embeddings)

            real_tokens_mask = (mlm_batch.attention_masks == 1)
            tokens_matched = ((decoded_tokens == mlm_batch.tokens_ids) & real_tokens_mask).sum()

            self.log( "tokens_matched", tokens_matched.item() )
            tokens_matched_accuracy = tokens_matched / real_tokens_mask.sum().item()
            self.log( "tokens_matched_accuracy",  tokens_matched_accuracy.item(), prog_bar=True )

            mask_token_id = self.tokenizer.token_to_id('[MASK]')
            self.log("num_decoded_masks", (decoded_tokens == mask_token_id).sum().item())

            unk_token_id = self.tokenizer.token_to_id('[UNK]')
            self.log("num_decoded_unkns", (decoded_tokens == unk_token_id).sum().item())

            return mlm_bertout, mlm_loss, decoded_tokens

        return mlm_bertout, mlm_loss



# copy-paste https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
def cli_main(args=None):

    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", required=False, type=str)
    parser.add_argument("--strict", default=False, action='store_true')
    parser.add_argument("--name", type=str, required=True)

    parser.add_argument("--early_stopping_monitor", type=str, default='tokens_matched_accuracy')
    parser.add_argument("--early_stopping_mode", type=str, default='max')
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    parser.add_argument('--tokenizer', help='path to pretrained tokenizer', type=str, required=True)
    parser.add_argument('--dataset', help='datasets dataset name', type=str, required=True)
    parser.add_argument('--languages', help='dataset languages to tokenize', type=str, required=True)

    dm_class = WMT20DataModule
    parser = dm_class.add_argparse_args(parser)
    parser = BertLightningModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args(args)

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = Tokenizer.from_file(args.tokenizer)
    # [UNK],[SEP],[PAD],[MASK],[ECHO],[TRANSLATE]
    special_tokens = [ '[UNK]', '[PAD]', '[TRANSLATE]', '[ECHO]', '[MASK]', '[SEP]', ]
    assert tokenizer.add_special_tokens(special_tokens) == 0, f'one of special tokens not in tokenizer: {special_tokens}'

    dm = dm_class.from_argparse_args(args, tokenizer=tokenizer, dataset=args.dataset, languages=args.languages, device='cuda')
    dm.setup()

    assert dm.device == 'cuda'

    if args.max_steps == -1:
        args.max_steps = None

    if args.checkpoint is not None:
        print("Restoring from checkpoint", args.checkpoint)
        bert_model = BertLightningModule.load_from_checkpoint(args.checkpoint, strict=args.strict)
        bert_model.hparams.noam_scaler = args.noam_scaler
        bert_model.hparams.lr= args.lr
        bert_model.hparams.noam_opt_warmup_steps = args.noam_opt_warmup_steps
        bert_model.hparams.scheduler = args.scheduler
        bert_model.hparams.scheduler_patience = args.scheduler_patience
        bert_model.hparams.noam_step_factor = args.noam_step_factor
        bert_model.tokenizer = tokenizer

    else:
        args_dict = vars(args)
        lightning_module_args = { k: args_dict[k] for k in args_dict.keys() if args_dict[k] is not None }
        lightning_module_args['tokenizer'] = tokenizer
        bert_model = BertLightningModule(**lightning_module_args)

    trainer_logger = pl.loggers.TensorBoardLogger("lightning_logs", name=args.name)
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=args.early_stopping_monitor,
        mode=args.early_stopping_mode,
        min_delta=args.early_stopping_min_delta,
        patience=args.early_stopping_patience,
        verbose=True,
    )
    trainer = pl.Trainer.from_argparse_args(args, logger=trainer_logger, callbacks=[early_stop_callback])
    trainer.fit(bert_model, datamodule=dm)
    return dm, bert_model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()



