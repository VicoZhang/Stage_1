import re
import shutil

import numpy as np
import random
import imageio
import os


def read_variable_name(f_path):
    L = {}
    f = open(f_path, 'r', encoding='utf-8')
    for line in f:
        reformat = re.compile(r"PGB\((?P<PGB>.*?)\)(.*?)"
                              r"(.*?)Desc=\"(?P<Desc>.*?)\"", re.S)
        index = reformat.finditer(line)
        for item in index:
            L[item.group("PGB")] = item.group("Desc").replace(":", "_")
    f.close()
    return L


def read_simulation_time(f_path):
    f = open(f_path, 'r', encoding='utf-8')
    L = []
    for line in f:
        line = line.replace("\n", '')
        line = re.split(r"[ ]+", line)
        if line[1]:
            L.append(line[1])
        else:
            continue
    f.close()
    return L


def read_variable_value(f_path, index):
    f = open(f_path, 'r', encoding='utf-8')
    data_temp = []
    for line in f:
        line = line.replace("\n", '')
        line = re.split(r"[ ]+", line)
        data_temp.append(line)
    f.close()
    column_number = list(index.keys())
    column_name = list(index.values())
    dir_temp = {}
    for i in range(len(column_name)):
        L = []
        for line in data_temp:
            if line[1]:
                L.append(eval(line[eval(column_number[i]) + 1]))
            else:
                continue
        dir_temp[column_name[i]] = L
    return dir_temp


def Mix_data(data):
    dic = {'FaultType': data['FaultType']}
    temp_U = np.array((data['U_1'], data['U_2'], data['U_3']))
    dic['U'] = temp_U.flatten('F')
    temp_I = np.array((data['I_1'], data['I_2'], data['I_3']))
    dic['I'] = temp_I.flatten('F')
    dic['TransitionR'] = data['TransitionR']
    dic['FaultTime'] = data['FaultTime']
    return dic


def Cut_data(data, times, length):
    for iii in range(len(times)):
        if eval(times[iii]) == data['FaultTime'][iii]:
            data['U'] = data['U'][int((iii - length / 6) * 3):int((iii + length - length / 6) * 3)]
            data['I'] = data['I'][int((iii - length / 6) * 3):int((iii + length - length / 6) * 3)]
            data['TransitionR'] = data['TransitionR'][iii]
            data['FaultTime'] = data['FaultTime'][iii]
            data['FaultType'] = data['FaultType'][iii]
            break
    return data


def turn_grayscale(data):  # data ?????????????????????
    lens = len(data)
    max_start = lens - dimension_grayscale ** 2
    starts = []
    gray_scales = []
    for i in range(sampling_value):
        while True:
            start = random.randint(0, max_start)
            if start not in starts:
                starts.append(start)
                break
        temp = data[start: start + dimension_grayscale ** 2]
        temp = np.array(temp)
        gray_temp = temp.reshape(dimension_grayscale, dimension_grayscale)
        gray_scales.append(gray_temp)
    return gray_scales


def save_npz(graydata, fault_place, fault_type):  # ??????????????????????????????
    global step_npz
    number = step_npz
    npz_save_path = os.path.join('..', 'Data', 'DataSet_npz', fault_place, fault_type)
    img_name = "{}_{}_{}".format(fault_place, fault_type, number)
    if os.path.exists(npz_save_path):
        np.savez(os.path.join(npz_save_path, img_name), *graydata)
    else:
        os.makedirs(npz_save_path)
        np.savez(os.path.join(npz_save_path, img_name), *graydata)
    return


def save_jpg(fault_place, fault_type):
    global step_npz
    global step_jpg
    npz_read_path = os.path.join('..', 'Data', 'DataSet_npz', fault_place, fault_type)
    npz_img_name = "{}_{}_{}".format(fault_place, fault_type, step_npz)

    with np.load(os.path.join(npz_read_path, npz_img_name + '.npz'), allow_pickle=True) as npz_file:
        for file in npz_file:
            save_path = os.path.join('..', 'Data', 'DataSet', fault_place, fault_type)
            jpg_img_name = "{}_{}_{}".format(fault_place, fault_type, step_jpg)
            temp = npz_file[file]
            temp = temp.astype(np.float64)
            if os.path.exists(save_path):
                imageio.imwrite(os.path.join(save_path, jpg_img_name + '.jpg'), temp)
                print("savejpg")
            else:
                os.makedirs(save_path)
                imageio.imwrite(os.path.join(save_path, jpg_img_name + '.jpg'), temp)
            step_jpg += 1
            step_npz += 1
    return


def draw_grayscale(data, value_name):
    for item in data:
        if item == value_name:
            temp = data[item]
            fault_type = str(int(data['FaultType']))
            fault_place = 'T1'
            temp = turn_grayscale(temp)
            save_npz(temp, fault_place, fault_type)
            save_jpg(fault_place, fault_type)
            print("save")
    delete_npz(npz_path)


def delete_npz(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    return


if __name__ == '__main__':
    root_path = '../Data/Initial_data'
    npz_path = '../Data/DataSet_npz'
    dimension_grayscale = 16
    sampling_value = 100
    sampling_length = 120
    step_npz = 0
    step_jpg = 0

    for ii in range(len(os.listdir(root_path))):  # ???????????????
        file_name = os.listdir(root_path)[ii]
        index_root_path = os.path.join(root_path, os.listdir(root_path)[ii], 'Index')  # ???????????????
        value_root_path = os.path.join(root_path, os.listdir(root_path)[ii], 'Value')  # ???????????????
        for jj in range(len(os.listdir(index_root_path))):  # ??????????????????
            index_path = os.path.join(index_root_path, os.listdir(index_root_path)[jj])
            value_path = os.path.join(value_root_path, os.listdir(value_root_path)[jj])
            index_dir = read_variable_name(index_path)  # ??????????????????????????????
            SimulationTime_list = read_simulation_time(value_path)  # ???????????????????????????
            SimulationValue_dir = read_variable_value(value_path, index_dir)  # ???????????????????????????
            SimulationValue_dir = Mix_data(SimulationValue_dir)  # ????????????????????????
            SimulationValue_dir = Cut_data(SimulationValue_dir, SimulationTime_list, sampling_length)  # ????????????
            draw_grayscale(SimulationValue_dir, value_name='U')  # ???????????????

    print('????????????')
    print('???????????????{}???'.format(step_jpg))
