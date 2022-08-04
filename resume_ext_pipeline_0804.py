'''
Creator : xiaofan
Input : a parsed resume txt
output : information json
'''
import os
import json
import torch
import argparse
from tqdm import tqdm,trange
from utils import filter_cls,filter_info
from transformers import BertTokenizer, BertForSequenceClassification, BartForConditionalGeneration
def read_json(file):return json.load(open(file))
def read_jsonl(file):return [json.loads(line) for line in open(file).readlines()]
def write_json(obj,file):json.dump(obj,open(file,'w'),indent=4,ensure_ascii=False)
def w2f(s,file):
    with open(file, 'a+') as f:
        f.write(s+"\n")

parser = argparse.ArgumentParser()
parser.add_argument("--current_path",default="/home/xiaofan/kwcodes/resume_extraction/pipeline",type=str)
parser.add_argument("--cls_model_path",default="/home/xiaofan/kwcodes/resume_extraction/BERT_CLS/models/bert_res_cls_merge",type=str)
parser.add_argument("--ext_model_path_pe",default="/home/xiaofan/kwcodes/resume_extraction/triple_generation/models/rebel_resume_per_edu",type=str)
parser.add_argument("--ext_model_path_wp", default="/home/xiaofan/kwcodes/resume_extraction/triple_generation/models/rebel_resume_work_proj",type=str)
parser.add_argument("--test_file_path", default="/home/xiaofan/kwcodes/resume_extraction/data/txt_sample/tmp",type=str)
parser.add_argument("--cuda_num", default=2, type=int)
parser.add_argument("--num_beams", default=2, type=int)
parser.add_argument("--version", default=0, type=int)
args = parser.parse_args()
os.chdir(args.current_path)

def find_all_pdf(pdir):
    pdfdir = []
    names = []
    for root,ds,fs in os.walk(pdir):
        for f in fs:
            if ".txt" in f:
                names.append(f[:-4])
                pdfdir.append(os.path.join(root,f))
    return pdfdir,names

device = torch.device(f"cuda:{args.cuda_num}")
cls_model_path = args.cls_model_path
ext_model_path_pe = args.ext_model_path_pe
ext_model_path_wp = args.ext_model_path_wp
# block cls model
cls_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
cls_model = BertForSequenceClassification.from_pretrained(cls_model_path)
cls_model = cls_model.to(device)

model_name = 'fnlp/bart-base-chinese'
tokenizer_kwargs = {
    "use_fast": True,
    "additional_special_tokens": ['<rel>', '<obj>', '<subj>'],
}
ext_tokenizer = BertTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
ext_model_pe = BartForConditionalGeneration.from_pretrained(ext_model_path_pe)
ext_model_pe.resize_token_embeddings(new_num_tokens=len(ext_tokenizer))
ext_model_pe = ext_model_pe.to(device)
ext_model_pe.eval()
ext_model_wp = BartForConditionalGeneration.from_pretrained(ext_model_path_wp)
ext_model_wp.resize_token_embeddings(new_num_tokens=len(ext_tokenizer))
ext_model_wp = ext_model_wp.to(device)
ext_model_wp.eval()
max_source_length = 512
max_target_length = 256
max_cut_length = 512

label2id = {"个人信息":0,"教育经历":1,"工作经历":2,"项目经历":3}
id2label = {v:k for k,v in label2id.items()}
version = "0"
from datetime import datetime
when = datetime.now().strftime("%Y%m_%d_%H_%M")
def cut(obj,sec):return [obj[i:i+sec] for i in range(0,len(obj),sec)]

p1dir,names = find_all_pdf(args.test_file_path) # test_file_path is a path filled with parsed txt
if not os.path.exists(f"./tmp/{when}/"):os.makedirs(f"./tmp/{when}/")
if not os.path.exists(f"./result/{when}/"):os.makedirs(f"./result/{when}/")
for i in range(len(p1dir)):
    txt_path = p1dir[i]
    txt_name = names[i]
    wenben = open(txt_path).read()
    # resume data in
    txt_data_0 = list([i.strip() for i in open(txt_path).readlines() if i])
    txt_data = txt_data_0
    # block segmentation
    out_dic = {"个人信息":[],"教育经历":[],"工作经历":[],"项目经历":[]}
    label_flag = -1
    for i in trange(len(txt_data)):
        cls_model.eval()
        with torch.no_grad():
            text = txt_data[i]
            inputs = cls_tokenizer(text,
                                    add_special_tokens=True, 
                                    return_tensors='pt')
            inputs.to(device)
            outputs = cls_model(**inputs)
            label = int(outputs.logits.argmax(dim=1))
            w2f(f"{text}\t{id2label[label]}",f"./tmp/{when}/cls_{txt_name}_{version}.txt")
            out_dic[id2label[label]].append(text)
    out_dic1 = filter_cls(out_dic)
    w2f(json.dumps(out_dic1,ensure_ascii=False),f"./tmp/{when}/cls_{txt_name}_{version}.json")
    # detail information extraction
    ext_data = []
    for k,v in out_dic1.items():
        if k == "个人信息" or k == "教育经历":
            ext_data.append(str(k)+": "+" ".join(v))
        else:
            j = str(k)+": "+" ".join(v)
            prefix = j[:6]
            if len(j) > max_cut_length:
                for i in cut(j,max_cut_length):
                    ext_data.append(prefix+i)
            else:ext_data.append(j)
    for i in trange(len(ext_data)):
        text = ext_data[i]
        inputs = ext_tokenizer(text, max_length=max_source_length, padding="longest", truncation=True, return_tensors="pt")
        inputs.to(device)
        params = {"decoder_start_token_id":0,"early_stopping":False,"no_repeat_ngram_size":0,"length_penalty": 0,"num_beams":args.num_beams,"use_cache":True}
        if text.startswith("个人信息") or text.startswith("教育经历"):
            out_id = ext_model_pe.generate(inputs["input_ids"], attention_mask = inputs["attention_mask"], max_length=256, **params)
        else:
            out_id = ext_model_wp.generate(inputs["input_ids"], attention_mask = inputs["attention_mask"], max_length=512, **params)
        out_text = ext_tokenizer.decode(out_id[0],clean_up_tokenization_spaces=True)
        out_text = out_text.replace(" ","").replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").replace("[UNK]", "")
        w2f(out_text,f"./tmp/{when}/ext_{txt_name}_{version}.txt")
        w2f(text+"\t"+out_text,f"./tmp/{when}/t_ext_{txt_name}_{version}.txt")

    post_in_path = f"./tmp/{when}/ext_{txt_name}_{version}.txt"
    post_out_path = f"./result/{when}/{txt_name}_{version}.json"

    tmp = [i.strip() for i in open(post_in_path).readlines() if i]
    dic = {} # result dic
    dic["文本"] = wenben
    for i in tmp:
        i = [i for i in i.split("<rel>") if i]
        for j in i:
            j = j.split("<obj>")
            if len(j) == 2:
                tmp = dic.get(j[0],[])
                tmp.append(j[1])
                dic[j[0]] = tmp
            else:
                print(j)
    dic = filter_info(dic)
    write_json(dic,post_out_path)