import numpy as np
import csv
import pickle

def generate_feature_label(filename, n_feature: int = 6):

    """
    function generate_feature_label(filename, n_feature)

    :param
        filename(str: 'rightlc', 'leftlc' or 'lk'): path of the .csv file
        n_features(int): how many features are kept, default: 6

    :return
        headings(list: n_features): feature categories
        features(array: n_samples * n_features): n_samples = n_sequence * time steps inside each sequence
        label(int array: n_samples,): each element(label: 0, 1, 2, ...) has the same index as features
        seq_range(array: n_sequence+1,): record index of each sequence in the feature vector

    """

# open .csv file, store all the data in rows(list)
    with open(filename+'.csv', newline='', encoding='utf-8-sig') as csvfile:
        # read .csv file
        reader = csv.reader(csvfile)
        # store data in rows
        rows = [row for row in reader]
    csvfile.close()
    #print("reading", filename, ".csv", "is successed")

# extract n_sequences, store in sequence+'i'
    # headings(list): categories of the data (omit veh_id)
    headings = rows[0][1:]
    # n_sequence(int): total number of sequences
    n_sequence = int(rows[-1][0])
    # initialize length
    length = np.zeros((n_sequence,), dtype=int)
    # veh_num(int): count the number of vehicles (sequences)
    veh_num = 1
    # globals()[]: a way to combine string and variables in a variable name
    seq = 'sequence' + str(1)
    globals()[seq] = []
    # store data in sequence/label_seq+'i', whose type is going to be converted from list to ndarray
    for row in rows[1:]:
        # store data which come from one sequence
        if row[0] == str(veh_num):
            globals()[seq].append(row[1:])
            # length(array: n_sequence,) counts the time steps in one sequence
            length[veh_num-1] = length[veh_num-1] + 1
        # detect changes on sequence (veh_id), initialize a new array and then write data
        else:
            globals()[seq] = np.array(globals()[seq], dtype=float)
            seq = 'sequence' + row[0]
            veh_num = veh_num + 1
            globals()[seq] = []
            globals()[seq].append(row[1:])
            length[veh_num-1] = length[veh_num-1] + 1
    # convert the type of sequence+'i' from list to array
    globals()[seq] = np.array(globals()[seq], dtype=float)

# extract features
    # seq_range(array: n_sequence+1,): each element represent the start row of each sequence in the feature vector
    seq_range = np.zeros((n_sequence + 1,), dtype=int)
    for i in range(1, np.size(length, 0) + 1):
        next_seq_start = np.array([np.sum(length[0:i])])
        seq_range[i] = next_seq_start
    # features(array: (n_sequences*T)*number of categories)
    features = np.empty((np.sum(length), 6))
    for i in range(n_sequence):
        seq = 'sequence' + str(i+1)
        features[seq_range[i]:seq_range[i+1]] = globals()[seq][:]

# label the features by states
    # initialize label(array: n_samples,)
    label = np.zeros((np.sum(length[:]), ), dtype=int)
    # label lane keeping features
    # features[3]: target_acc_y, min(features[3,seq:seq+15])>0.4: acc is higher than 0.4 m/s^2 for 1.5 second in a row
    # 0: normal driving; 1: acceleration; 2: deceleration
    if 'lk' in filename:
        # lk class only contains one state
        label[:] = 0
        """
        # lk class contains three state that categorized by longitudinal acceleration
        for seq in range(np.sum(length)):
            if min(features[seq:seq+15, 3]) > 0.4:
                label[seq:seq+15] = 1
            elif max(features[seq:seq+15, 3]) < -0.4:
                label[seq:seq+15] = 2
        """
    # label the left lane changes
    # features[4]: ref_heading < -0.2, features[5]: yaw_rate <0/>0
    # 0: normal driving; 1: steer to left; 2: steer back
    elif 'leftlc' in filename:
        seq = 0
        while seq < np.sum(length):
            n = 1
            # heading angle is larger than 0.2
            while max(features[seq:seq+n, 4]) < -0.2 and max(features[seq:seq+n, 5]) < 0 and seq+n < np.sum(length):
                n = n + 1
            # the features all belong to sequence i
            for i in range(np.size(length, 0)):
                if seq_range[i] <= seq < seq_range[i+1]:
                    break
            # if the heading angle is less than 0.2 for 0.6 second in a row, this period will be labelled as steer
            if n > 6 and seq >= seq_range[i] + 10:
                if seq + n >= seq_range[i + 1]:
                    n = seq_range[i + 1] - 1 - seq
                if n == 0:
                    seq = seq + 1
                else:
                    label[seq:seq+n] = 1
                    seq = seq + n
            else:
                seq = seq + 1
        seq = 0
        while seq < np.sum(length):
            n = 1
            # heading angle is larger than 0.2
            while max(features[seq:seq+n, 4]) < -0.2 and min(features[seq:seq+n, 5]) > 0 and seq+n < np.sum(length):
                n = n + 1
            # the features all belong to sequence i
            # if the heading angle is less than 0.2 for 0.6 second in a row, this period will be labelled as steer back
            if n > 6:
                for i in range(np.size(length, 0)):
                    if seq_range[i] <= seq < seq_range[i + 1]:
                        if seq + n >= seq_range[i + 1]:
                            n = seq_range[i + 1] - 1 - seq
                        break
                if n == 0:
                    seq = seq+1
                else:
                    label[seq:seq+n] = 2
                    seq = seq + n
            else:
                seq = seq + 1
    # label the right lane changes
    # features[4]: ref_heading > 0.2, features[5]: yaw_rate >0/<0
    # 0: normal driving; 1: steer to right; 2: steer back
    elif 'rightlc' in filename:
        seq = 0
        while seq < np.sum(length):
            n = 1
            # heading angle is larger than 0.2
            while min(features[seq:seq+n, 4]) > 0.2 and min(features[seq:seq+n, 5]) > 0 and seq+n < np.sum(length):
                n = n + 1
            # the features all belong to sequence i
            for i in range(np.size(length, 0)):
                if seq_range[i] <= seq < seq_range[i+1]:
                    break
            # if the heading angle is less than 0.2 for 0.6 second in a row, this period will be labelled as steer
            if n > 6 and seq >= seq_range[i] + 10:
                if seq + n >= seq_range[i + 1]:
                    n = seq_range[i + 1] - 1 - seq
                if n == 0:
                    seq = seq + 1
                else:
                    label[seq:seq+n] = 1
                    seq = seq + n
            else:
                seq = seq + 1
        seq = 0
        while seq < np.sum(length):
            n = 1
            # heading angle is larger than 0.2
            while min(features[seq:seq+n, 4]) > 0.2 and max(features[seq:seq+n, 5]) < 0 and seq+n < np.sum(length):
                n = n + 1
            # the features all belong to sequence i
            # if the heading angle is less than 0.2 for 0.6 second in a row, this period will be labelled as steer back
            if n > 6:
                for i in range(np.size(length, 0)):
                    if seq_range[i] <= seq < seq_range[i+1]:
                        if seq + n >= seq_range[i + 1]:
                            n = seq_range[i + 1] - 1 - seq
                        break
                if n == 0:
                    seq = seq+1
                else:
                    label[seq:seq+n] = 2
                    seq = seq + n
            else:
                seq = seq + 1

# delete useless features
    features = np.delete(features, np.s_[0:6-n_feature], 1)
    del headings[0:6-n_feature]

# return value
    #print("features and labels are generated")
    return features, label, seq_range
