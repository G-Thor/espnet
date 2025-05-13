#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_fft=2048
n_shift=300
win_length=1200

base_model_dir=exp/tts_train_xvector_tacotron2_raw_phn_tacotron_g2p_en_no_space
base_model=${base_model_dir}/train.loss.ave_5best.pth

opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=my_training_set_phn
valid_set=my_dev_set_phn
test_sets="my_dev_set_phn my_eval_set_phn"

# train_config=conf/train.yaml
train_config=conf/tuning/finetune_xvector_tacotron2.yaml
# inference_config=conf/decode.yaml
inference_config=conf/tuning/decode_tacotron2.yaml
# optionally remove this to use the Griffin-Lim algorithm for vocoding
inference_args="--vocoder_tag=parallel_wavegan/vctk_style_melgan.v1"

# g2p=g2p_en_no_space # Include word separator
g2p=none # Include no word separator

# --train_args "--init_param ${base_model_dir}:tts:tts" \
./tts.sh \
    --lang en \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --cleaner tacotron \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --use_spk_embed true \
    --inference_args "${inference_args}" \
    --train_args "--init_param ${base_model_dir}:tts:tts" \
    --tag "finetune_phn" \
    --stage 2 \
    --stop_stage 5 \
    ${opts} "$@"

pyscripts/utils/make_token_list_from_config.py ${base_model_dir}/config.yaml
cp ${base_model_dir}/tokens.txt dump/token_list/phn_tacotron_none/tokens.txt


# stage 6: Fine-tune the model
./tts.sh \
    --lang en \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --cleaner tacotron \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --use_spk_embed true \
    --inference_args "${inference_args}" \
    --train_args "--init_param ${base_model}:tts:tts" \
    --tag "finetune_phn" \
    --stage 6 \
    ${opts} "$@"
