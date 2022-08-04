# resume_infomation_extraction_xfan
简历信息抽取pipeline

终端运行
sh run_pipeline.sh
即可

python resume_ext_pipeline_0804.py \
    --current_path=/home/xiaofan/kwcodes/resume_extraction/pipeline \
    --cls_model_path=/home/xiaofan/kwcodes/resume_extraction/BERT_CLS/models/bert_res_cls_merge \
    --ext_model_path_pe=/home/xiaofan/kwcodes/resume_extraction/triple_generation/models/rebel_resume_per_edu \
    --ext_model_path_wp=/home/xiaofan/kwcodes/resume_extraction/triple_generation/models/rebel_resume_work_proj \
    --test_file_path=/home/xiaofan/kwcodes/resume_extraction/data/txt_sample/tmp \
    --cuda_num=2 \
    --num_beams=2 \
    --version=0
