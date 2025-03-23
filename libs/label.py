def possible_qstate(bit_num, labels, label_type):
    d = {}
    
    if label_type == "compressed":
        for i in range(2**bit_num):
            binary_str = bin(i)[2:] # 2進数に変換：0bxx
            if i < len(labels):
                label = labels[i]
            else:
                label = -1
            d[binary_str.zfill(bit_num)] = label
            
    elif label_type == "one_hot":
        for label in range(2**bit_num):
            binary_str = bin(label)[2:] # 2進数に変換：0bxx
            d[binary_str.zfill(bit_num)[::-1]] = -1 # -1で初期化
        for label in range(bit_num):
            key = [0]*(bit_num)
            key[label] = 1
            d["".join(map(str, key[::-1]))] = labels[label]
    return d
