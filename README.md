# HowtoReadPaper
https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf


#### References:
* bert https://arxiv.org/pdf/1810.04805.pdf
* attention is all you need https://arxiv.org/pdf/1706.03762.pdf
* https://huggingface.co/transformers/

#### Attempts on BERT nmt
* https://slator.com/machine-translation/does-googles-bert-matter-in-machine-translation/
* https://arxiv.org/abs/1909.12744
* https://arxiv.org/abs/1905.02450

#### Unsupervised NMT
* https://www.aclweb.org/anthology/P19-1019.pdf
* https://arxiv.org/pdf/1804.07755.pdf
* https://arxiv.org/pdf/2001.08210.pdf

#### Другие походы к улучшению качества перевода

##### Context-Aware Monolingual Repair for Neural Machine Translation
https://arxiv.org/pdf/1909.01383.pdf
Исправлялка на одном языке и не только

##### A Study with Machine Translation and Language Modeling Objectives
https://arxiv.org/pdf/1909.01380.pdf
В разных задачах разное представление эмбэддингов токенов


##### Данные для перевода можно спереть отсюда
https://github.com/lena-voita/good-translation-wrong-in-context#training-data


* обучаем по словарю
* scaling up transformers like efficient nets?
* Recurrent Bert разное количество слоев

Adversarial NMT -- может быть хорошо применено к прогрессивному переводу
* https://arxiv.org/pdf/1905.11946v5.pdf

non autoregressive nmt
* https://arxiv.org/pdf/1711.02281.pdf
* https://www.aclweb.org/anthology/2020.autosimtrans-1.4/
* https://openreview.net/forum?id=wOI9hqkvu_



Прунинг неавторегрессионных моделей


Related Works:

Гитхаб со всеми работами по NA https://github.com/kahne/NonAutoregGenProgress


Gu первое исследование NAT + Fertility prediction
Кач-во перевода немного хуже, но скорость больше.
https://arxiv.org/pdf/1711.02281.pdf
https://github.com/salesforce/nonauto-nmt/blob/master/model.py

Jason Lee, iterative refinement + denoising autoencoders
https://www.aclweb.org/anthology/D18-1149.pdf
https://github.com/nyu-dl/dl4mt-nonauto/tree/multigpu


Yu Bao, PNAT Position Learning
недавняя работа с добавлением position wise enbeddings
с хорошим related works
https://openreview.net/pdf?id=BJe932EYwS


FlowSeq
Латентные переменные с помощью [Normalizing Flows](https://arxiv.org/pdf/1505.05770.pdf)
https://arxiv.org/pdf/1909.02480.pdf


Deep Encoder, Shallow Decoder
https://openreview.net/pdf?id=KpfasTaLUpq
https://github.com/jungokasai/deep-shallow

Говорят, что авторегорессионные модели быстрее на больших размерах батча.
Мб это из-за того, что там нелинейное внимание?

Но сравниваются только с итеративными вариантами NAT s2s.
И еще есть идея, что в декодере для NAT можно не использовать
несколько аттеншнов, а чередовать на разных слоях только
аттеншн одного типа.

### Лосс-функции

Можно исследовать как в зависисмости от лосс ф-ии будет меняться
внутренние слои трансформена

Order-Agnostic CrossEnthropyLoss
https://arxiv.org/pdf/2106.05093.pdf


Aligned Cross Entropy
https://arxiv.org/abs/2004.01655



# Гипотезы и идеи:

### из https://www.aclweb.org/anthology/P19-1580.pdf
* исследовать как прунятся non autoregressive transformers
* добавить распределение голов по функциям

### из source_target_contributions_to_nmt
https://lena-voita.github.io/posts/source_target_contributions_to_nmt.html
* сделать LRP как у Воуты для анализа, какие части предложения влияют

### исследовать поведение для разных задач
* MT, LM, MLM, как в https://arxiv.org/pdf/1909.01380.pdf

Проблемы:
* повторение слов

Можно попробовать модицицировать лосс, чтобы вместо повторений
предсказывались <EMPTY> токены и их фильтровать из конечного предложения.
И сделать так, чтобы за них не штрафовали модель, но сильно много таких токенов
тоже нельзя допускать.








-----------

Неструктруированые todo

### linformer
https://github.com/pytorch/fairseq/tree/master/examples/linformer
https://arxiv.org/pdf/2006.04768.pdf

### wav2vec
https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md
https://arxiv.org/abs/2006.11477

### nar
https://github.com/pytorch/fairseq/blob/master/examples/nonautoregressive_translation/README.md

### Levenshtein Transformer
https://arxiv.org/pdf/1905.11006.pdf

### UNDERSTANDING KNOWLEDGE DISTILLATION IN NON-AUTOREGRESSIVE MACHINE TRANSLATION
https://arxiv.org/pdf/1911.02727.pdf

-----------

todo Просто интересно

Fine-Tuning by Curriculum Learning for Non-Autoregressive
https://arxiv.org/pdf/1911.08717.pdf

-----------
