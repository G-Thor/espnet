. path.sh
. cmd.sh
. utils/parse_options.sh

python3 -m espnet2.bin.tts_inference \
        --ngpu 1 \
        --data_path_and_name_and_type dump/raw/my_eval_set/text,text,text \
        --data_path_and_name_and_type dump/raw/my_eval_set/wav.scp,speech,sound \
        --model_file exp/tts_train_xvector_tacotron2_raw_phn_tacotron_g2p_en_no_space/train.loss.ave_5best.pth \
        --train_config exp/tts_train_xvector_tacotron2_raw_phn_tacotron_g2p_en_no_space/config.yaml \
        --output_dir pretrained_outputs \
        --data_path_and_name_and_type dump/espnet_spk/my_eval_set/espnet_spk.scp,spembs,kaldi_ark \
        --config conf/tuning/decode_tacotron2.yaml \
        --vocoder_tag parallel_wavegan/vctk_hifigan.v1


# python3 -m espnet2.bin.tts_inference \
#         --ngpu 1 \
#         --data_path_and_name_and_type dump/raw/dev/text,text,text \
#         --data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound \
#         --key_file exp/tts_train_xvector_tacotron2_raw_phn_tacotron_g2p_en_no_space/decode_tacotron2_vocoder_tagparallel_wavegan/vctk_hifigan.v1_train.loss.ave/dev/log/keys.1.scp \
#         --model_tag kan-bayashi/vctk_tts_train_xvector_tacotron2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave \
#         --output_dir pretrained_outputs_val \
#         --vocoder_file none \
#         --data_path_and_name_and_type dump/xvector/dev/xvector.scp,spembs,kaldi_ark \
#         --config conf/tuning/decode_tacotron2.yaml \
#         --vocoder_tag parallel_wavegan/vctk_hifigan.v1

# python3 -m espnet2.bin.tts_inference --ngpu 1 --data_path_and_name_and_type dump/raw/dev/text,text,text \
# --data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound \
# --key_file exp/tts_train_xvector_tacotron2_raw_phn_tacotron_g2p_en_no_space/decode_tacotron2_vocoder_tagparallel_wavegan/vctk_hifigan.v1_train.loss.ave/dev/log/keys.1.scp \
# --model_file exp/tts_train_xvector_tacotron2_raw_phn_tacotron_g2p_en_no_space/train.loss.ave.pth \
# --train_config exp/tts_train_xvector_tacotron2_raw_phn_tacotron_g2p_en_no_space/config.yaml \
# --output_dir exp/tts_train_xvector_tacotron2_raw_phn_tacotron_g2p_en_no_space/decode_tacotron2_vocoder_tagparallel_wavegan/vctk_hifigan.v1_train.loss.ave/dev/log/output.1 --vocoder_file none --config conf/tuning/decode_tacotron2.yaml \
# --data_path_and_name_and_type dump/espnet_spk/dev/espnet_spk.scp,spembs,kaldi_ark \
# --vocoder_tag parallel_wavegan/vctk_hifigan.v1 