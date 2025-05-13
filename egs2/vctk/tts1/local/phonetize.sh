#!/usr/bin/env bash

. ./cmd.sh

set -euo pipefail

for dset in my_training_set my_eval_set my_dev_set; do
    ./utils/copy_data_dir.sh ./data/"${dset}"{,_phn};
    ./pyscripts/utils/convert_text_to_phn.py --nj 1 --g2p g2p_en_no_space ./data/"${dset}"{,_phn}/text;
done