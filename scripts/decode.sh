#!/usr/bin/env bash

python tensor2tensor/bin/t2t_decoder.py \
--data_dir=t2t_data/uschema_words_freebase/   \
--problems=translate_uschema_freebase   \
--model=transformer   \
--hparams_set=transformer_base  \
--output_dir=t2t_train/uschema_freebase/ \
--worker_gpu=1 \
--decode_from_file /iesl/data/clueweb_2016_full/freebase_ep_subset/2500k_ep_subset/dev.lang1 \
--decode_to_file new.tmp