# resume_infomation_extraction_xfan  
简历信息抽取pipeline  

终端运行  
sh run_pipeline.sh  
即可  

模型和test_file文件夹可以使用绝对路径也可放在当前文件夹下使用相对路径  
输入为含有简历parsed文本txt文件的文件夹，代码自动搜索文件夹中所有txt进行推理  
输出在results路径中，为json格式  
tmp文件夹中为模型中间输出结果，用于调试  
  
python resume_ext_pipeline_0804.py \
    --current_path=/home/xiaofan/resume_extraction/pipeline \ # 当前pipeline文件夹绝对路径  
    --cls_model_path=/home/xiaofan/resume_extraction/BERT_CLS/models/bert_res_cls_merge \ # 分类模型绝对路径  
    --ext_model_path_pe=/home/xiaofan/resume_extraction/triple_generation/models/rebel_resume_per_edu \ # 抽取模型1绝对路径  
    --ext_model_path_wp=/home/xiaofan/resume_extraction/triple_generation/models/rebel_resume_work_proj \ # 抽取模型2绝对路径  
    --test_file_path=/home/xiaofan/resume_extraction/data/txt_sample/tmp \ # 输入的文件夹，将已经parse成txt的简历文件们放在该文件夹路径下即可  
    --cuda_num=2 \ # 使用的GPU号数  
    --num_beams=2 \ # 超参，生成式抽取模型使用beam search，这个数值越高，推理越慢，结果可能越准确  
    --version=0 # 会输出在结果文件的文件名中，用于输出版本控制和对照观察  
