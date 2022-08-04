import re

def filter_cls(dic):
    for i in dic["个人信息"]:
        if "项目" in i:
            dic["项目经历"].append(i)
            dic["个人信息"].remove(i)
        elif "工作" in i:
            dic["工作经历"].append(i)
            dic["个人信息"].remove(i)
        elif "教育" in i:
            dic["教育经历"].append(i)
            dic["个人信息"].remove(i)
    for i in dic["教育经历"]:
        if "项目" in i:
            dic["项目经历"].append(i)
            dic["教育经历"].remove(i)
        elif "工作" in i:
            dic["工作经历"].append(i)
            dic["教育经历"].remove(i)
    return dic

def filter_info(dic):
    dic["filter_info"] = []
    if "电话号码" in dic:
        dic["电话号码"] = list(set(dic["电话号码"]))
        for i in dic["电话号码"]:
            if not i.isdigit():
                dic["filter_info"].append(i)
                dic["电话号码"].remove(i)
    if "邮箱" in dic:
        dic["邮箱"] = list(set(dic["邮箱"]))
        for i in dic["邮箱"]:
            if "@" not in i:
                dic["filter_info"].append(i)
                dic["邮箱"].remove(i)
    if "学校" in dic:
        dic["学校"] = list(set(dic["学校"]))
        for i in dic["学校"]:
            if "学" not in i:
                dic["filter_info"].append(i)
                dic["学校"].remove(i)
    if "所在公司" in dic:
        dic["所在公司"] = list(set(dic["所在公司"]))
        for i in dic["所在公司"]:
            if "公司" not in i:
                dic["filter_info"].append(i)
                dic["所在公司"].remove(i)
    if "性别" in dic:
        dic["性别"] = list(set(dic["性别"]))
        for i in dic["性别"]:
            if i != "男" or i != "女":
                dic["filter_info"].append(i)
                dic["性别"].remove(i)
    return dic