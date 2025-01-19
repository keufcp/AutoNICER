import os
import requests
# import openpyxl
import glob
import shutil
import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Variable


def calcurate_median(pred_score):

    # ソート済みの配列を生成
    sorted_arr = np.sort(pred_score)

    # 要素数が奇数か偶数かをチェック
    if len(sorted_arr) % 2 == 1:
        # 奇数の場合は厳密な中央値を計算
        median_value = np.median(sorted_arr)
        median_indices = np.where(pred_score == median_value)[0]
    else:
        # 偶数の場合は中央の2つからランダムに選ぶ
        mid_idx1 = len(sorted_arr) // 2 - 1  # より小さい方
        mid_idx2 = len(sorted_arr) // 2      # より大きい方

        # ランダムに1つ選ぶ
        chosen_mid_idx = random.choice([mid_idx1, mid_idx2])
        median_value = sorted_arr[chosen_mid_idx]
        
        # インデックスを元の配列で検索
        median_indices = np.where(pred_score == median_value)[0][0]
    
    return median_value, median_indices


def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename

def make_csv(opt):

    with open(opt.path_to_save_csv+"/test.csv", "w") as file:
        file.write("image_id,score2,score3,score4,score5,score6,score7,score8,score9,score10,score11,tag1,tag2,tag3\n")

    img_list = glob.glob(opt.path_to_images+"/1/*.jpg")
    with open(opt.path_to_save_csv+"/test.csv", "a") as file:
        for i in range(1, len(img_list)+1):
            file.write(os.path.splitext(os.path.basename(img_list[i-1]))[0]+",1,1,1,1,1,1,1,1,1,1,1,1,1\n")

def save_excel(opt, pred_score):

    if os.path.isfile(opt.excel_path):
        wb = openpyxl.load_workbook(opt.excel_path)
    else:
        wb = openpyxl.Workbook()

    # for sheet_name in ["result", "aesthetic", "saliency"]:
    if "result" in wb.sheetnames:
        ws = wb["result"]
    else:
        ws = wb.create_sheet(title="result")
        names = ["画像名", "審美性スコア"]
        for i, name in enumerate(names, 1):
            ws.cell(1, i).value = name
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])
    
    img_list = glob.glob(opt.path_to_images+"/*.jpg")
    for i, pred_score_one in enumerate(pred_score, 2):
        ws.cell(i, 1).value = os.path.splitext(os.path.basename(img_list[i-2]))[0]
        ws.cell(i, 2).value = pred_score_one

    wb.save(opt.excel_path)

def save_dataset(opt, pred_score, i, j):

    max_score = np.max(pred_score)
    max_index = np.argmax(pred_score) + 1
    high_img_path = opt.path_to_images+"/"+str(j-1)+"/"+str(max_index)+".jpg"
    high_save_path = opt.result_path+"/high/"+str(i)+".jpg"
    shutil.copyfile(high_img_path, high_save_path)

    min_score = np.min(pred_score)
    min_index = np.argmin(pred_score) + 1
    low_img_path = opt.path_to_images+"/"+str(j-1)+"/"+str(min_index)+".jpg"
    low_save_path = opt.result_path+"/low/"+str(i)+".jpg"
    shutil.copyfile(low_img_path, low_save_path)

    middle_score, middle_index = calcurate_median(pred_score)
    middle_img_path = opt.path_to_images+"/"+str(j-1)+"/"+str(middle_index)+".jpg"
    middle_save_path = opt.result_path+"/middle/"+str(i)+".jpg"
    shutil.copyfile(middle_img_path, middle_save_path)

    return high_save_path, max_score, low_save_path, min_score, middle_save_path, middle_score

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()