#!/usr/bin/env bash

python tensor2tensor/bin/t2t_trainer.py   \
--generate_data   \
--data_dir t2t_data/uschema_medium   \
--problems=translate_uschema_freebase_medium   \
--model=transformer   \
--hparams_set=transformer_base_single_gpu \
--hparams='batch_size=1024' \
--output_dir t2t_train/uschema_medium \
--worker_gpu=4 \
--local_eval_frequency 10000 \
--keep_checkpoint_max=3 \
--eval_steps 1
