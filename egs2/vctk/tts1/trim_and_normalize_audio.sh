#!/bin/bash
wav_path=/data/tts/vctk/VCTK-Corpus/wav48/p374

. path.sh
. cmd.sh
python trim_and_normalize_audio.py \
    --input_dir "${wav_path}" \
    --output_dir normalized_audio \
    --threshold -20