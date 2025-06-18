import os
import sys

block_size = 5
t = 4 # threshold

def Binary_block(L, start,last):
    QFs = 0
    block = block_size
    start_idx = start
    #print('start_idx', start_idx)
    last_idx = last
    #print('last_idx', last_idx)
    mid = int((start_idx + last_idx + 1) / 2)
    #print('mid', mid)
    block_R = sum(L[mid-1 : mid + block-1])
    #print('block_R', block_R)
    block_L = sum(L[mid - block-1 : mid-1])
    #print('block_L', block_L)
    if block_R != 0:   # right
        if block_R > t:
            Binary_bloack(L, mid, last_idx)
        else:
            if block_L == block_size:
                for j in range(0, 5, 1):
                    if block_R == 1:  # case 1   10000
                        if L[mid + j - 1] == 1:
                            QFs = mid + j
                            print('QFs', QFs)
                            break
                    else:
                        if L[mid + j-1] == 0 and L[mid + j] == 1:  # case 2   00111
                            QFs = mid + j+1
                            print('QFs', QFs)
                            break
                        if L[mid + j-1] == 1 and L[mid + j] == 0:  # case 2   11100
                            QFs = mid + j
                            print('QFs', QFs)
                            break
            else:
                tempt1 = 0
                tempt2 = 0
                for j in range(0, 5, 1):
                    if L[mid + j -1] == 1:
                        #print('data1', mid + j)
                        tempt1 = mid + j
                        break
                for i in range(0, 5, 1):
                    if L[mid - i -1] == 1:
                        #print('data2', mid - j)
                        tempt2 = mid - i
                        break
                QFs = int((tempt1 + tempt2)/2)
                print('QFs', QFs)

    else:
        if block_L < t:   #  left
            Binary_bloack(L, start_idx, mid)
        else:
            if block_L == block_size:
                    QFs = mid - 1
                    print('QFs', QFs)
            else:
                for j in range(0, 5, 1):
                    if L[mid + j - 1] == 0 and L[mid + j] == 1:  # case 1   01111
                        QFs = mid + j + 1
                        print('QFs', QFs)
                        break
                    if L[mid + j - 1] == 1 and L[mid + j] == 0:  # case 2   11110
                        QFs = mid + j
                        print('QFs', QFs)
                        break


if __name__ == '__main__':
    QF = 0
    for i in range(1, 50, 1):
        txt_tables = []
        SQF_DATA_DIR = 'G:/JND/data_record/first_jnd/'
        txt_file = open(SQF_DATA_DIR + 'result_image' + str(i) + '.txt', 'r')
        lines = txt_file.readline() # 读取第一行
        while lines:
            jnd_pos_seq_str = lines.lstrip().rstrip().split(',')
            jnd_pos_seq = [i for i in jnd_pos_seq_str]
            JND_first_point = int(jnd_pos_seq[1])
            JND = jnd_pos_seq[0]
            #print(JND, JND_first_point)
            txt_tables.append(JND_first_point)
            lines = txt_file.readline()
        #print(txt_tables)
        label = txt_tables
        del(txt_tables)
        left = 0
        right = len(label)
        QF = Binary_block(label, left, right)
