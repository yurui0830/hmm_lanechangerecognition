import numpy as np
import csv


def concatenate_feature_label(filename, n_feature: int = 6, satruation: bool = False):

    """
    function concatenate_feature_label(filename, n_feature)

    :param
        filename(str: 'rightlc', 'leftlc' or 'lk'): filename of the .csv file
        n_features(int): how many features will be kept, default: 6
        saturation(bool): if set a saturation value for heading angle and yaw rate or not, default: True

    :return
        features(array: n_samples * n_features): n_samples = n_sequence * time steps inside each sequence
        label(int array: n_samples,): each element(label: 0, 1, 2, ...) has the same index as features
        seq_range(array: n_sequence+1,): record index of each sequence in the feature vector
    """

# open .csv file, store all the data in rows(list)
    with open(filename+'_101.csv', newline='', encoding='utf-8-sig') as csvfile:
        # read .csv file
        reader_1 = csv.reader(csvfile)
        # store data in rows
        rows_1 = [row for row in reader_1]
    csvfile.close()
    with open(filename+'_i80.csv', newline='', encoding='utf-8-sig') as csvfile:
        # read .csv file
        reader_2 = csv.reader(csvfile)
        # store data in rows
        rows_2 = [row for row in reader_2]
    csvfile.close()

# extract n_sequences, store in sequence+'i'
    # headings(list): categories of the data (omit veh_id)
    headings = rows_1[0][1:]
    # n_sequence(int): total number of sequences
    n_sequence_1 = int(rows_1[-1][0])
    n_sequence_2 = int(rows_2[-1][0])
    # initialize length
    length_1 = np.zeros((n_sequence_1,), dtype=int)
    length_2 = np.zeros((n_sequence_2,), dtype=int)
# extract n_sequence_1 from file_1
    # veh_num(int): count the number of vehicles (sequences)
    veh_num = 1
    # globals()[]: a way to combine string and variables in a variable name
    seq = 'sequence_1_' + str(1)
    globals()[seq] = []
    # store data in sequence/label_seq+'i', whose type is going to be converted from list to ndarray
    for row in rows_1[1:]:
        # store data which come from one sequence
        if row[0] == str(veh_num):
            globals()[seq].append(row[1:])
            # length(array: n_sequence,) counts the time steps in one sequence
            length_1[veh_num-1] = length_1[veh_num-1] + 1
        # detect changes on sequence (veh_id), initialize a new array and then write data
        else:
            globals()[seq] = np.array(globals()[seq], dtype=float)
            seq = 'sequence_1_' + row[0]
            veh_num = veh_num + 1
            globals()[seq] = []
            globals()[seq].append(row[1:])
            length_1[veh_num-1] = length_1[veh_num-1] + 1
    # convert the type of sequence+'i' from list to array
    globals()[seq] = np.array(globals()[seq], dtype=float)
# extract n_sequence_2 from file_2
    # veh_num(int): count the number of vehicles (sequences)
    veh_num = 1
    # globals()[]: a way to combine string and variables in a variable name
    seq = 'sequence_2_' + str(1)
    globals()[seq] = []
    # store data in sequence/label_seq+'i', whose type is going to be converted from list to ndarray
    for row in rows_2[1:]:
        # store data which come from one sequence
        if row[0] == str(veh_num):
            globals()[seq].append(row[1:])
            # length(array: n_sequence,) counts the time steps in one sequence
            length_2[veh_num - 1] = length_2[veh_num - 1] + 1
        # detect changes on sequence (veh_id), initialize a new array and then write data
        else:
            globals()[seq] = np.array(globals()[seq], dtype=float)
            seq = 'sequence_2_' + row[0]
            veh_num = veh_num + 1
            globals()[seq] = []
            globals()[seq].append(row[1:])
            length_2[veh_num - 1] = length_2[veh_num - 1] + 1
    # convert the type of sequence+'i' from list to array
    globals()[seq] = np.array(globals()[seq], dtype=float)

# extract features
    # features(array: number of total sequences*number of categories)
    features = np.empty((np.sum(length_1)+np.sum(length_2), 6))
    n_sequence = n_sequence_1 + n_sequence_2
    length = np.zeros((n_sequence,), dtype=int)
    # seq_range(array: n_sequence+1,): each element represent the start row of each sequence in the feature vector
    seq_range = np.zeros((n_sequence + 1,), dtype=int)
    i_1 = i_2 = 1
    for i in range(1, np.size(length, 0)+1):
        if i % 2 == 1 and i_1 <= n_sequence_1:
            seq_1 = 'sequence_1_' + str(i_1)
            features[seq_range[i-1]:seq_range[i-1]+length_1[i_1-1]] = globals()[seq_1][:]
            length[i-1] = length_1[i_1-1]
            i_1 = i_1 + 1
        elif i % 2 == 0 and i_2 <= n_sequence_2:
            seq_2 = 'sequence_2_' + str(i_2)
            features[seq_range[i-1]:seq_range[i-1]+length_2[i_2-1]] = globals()[seq_2][:]
            length[i-1] = length_2[i_2-1]
            i_2 = i_2 + 1
        elif i % 2 == 1 and i_1 > n_sequence_1:
            seq_2 = 'sequence_2_' + str(i_2)
            features[seq_range[i-1]:seq_range[i-1]+length_2[i_2-1]] = globals()[seq_2][:]
            length[i-1] = length_2[i_2-1]
            i_2 = i_2 + 1
        elif i % 2 == 0 and i_2 > n_sequence_2:
            seq_1 = 'sequence_1_' + str(i_1)
            features[seq_range[i-1]:seq_range[i-1]+length_1[i_1-1]] = globals()[seq_1][:]
            length[i-1] = length_1[i_1-1]
            i_1 = i_1 + 1
        seq_range[i] = np.array([np.sum(length[:i])])

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

# set saturation values for heading angle and yaw rate
    if satruation == 1:
        for i in range(np.size(features,0)):
            if features[i,4] > 2.5:
                features[i,4] = 2.5
            elif features[i,4] < -2.5:
                features[i,4] = -2.5
            if features[i,5] > 0.8:
                features[i,5] = 0.8
            elif features[i,5] < -0.8:
                features[i, 5] = -0.8

# delete useless features
    if n_feature == 4:
        features = np.delete(features, [1, 3], 1)
    elif n_feature == 3:
        features = np.delete(features, [0, 1, 3], 1)

# return value
    return features, label, seq_range



def feature_label(filename, dataset, n_feature: int = 6, satruation: bool = False):

    """
    function concatenate_feature_label(filename, n_feature)

    :param
        filename(str: 'rightlc', 'leftlc' or 'lk'): filename of the .csv file
        n_features(int): how many features will be kept, default: 6
        saturation(bool): if set a saturation value for heading angle and yaw rate or not, default: True

    :return
        features(array: n_samples * n_features): n_samples = n_sequence * time steps inside each sequence
        label(int array: n_samples,): each element(label: 0, 1, 2, ...) has the same index as features
        seq_range(array: n_sequence+1,): record index of each sequence in the feature vector
    """

# open .csv file, store all the data in rows(list)
    with open(filename+'_'+dataset+'.csv', newline='', encoding='utf-8-sig') as csvfile:
        # read .csv file
        reader_1 = csv.reader(csvfile)
        # store data in rows
        rows_1 = [row for row in reader_1]
    csvfile.close()

# extract n_sequences, store in sequence+'i'
    # headings(list): categories of the data (omit veh_id)
    headings = rows_1[0][1:]
    # n_sequence(int): total number of sequences
    n_sequence = int(rows_1[-1][0])
    # initialize length
    length = np.zeros((n_sequence,), dtype=int)
# extract n_sequence_1 from file_1
    # veh_num(int): count the number of vehicles (sequences)
    veh_num = 1
    # globals()[]: a way to combine string and variables in a variable name
    seq = 'sequence_' + str(1)
    globals()[seq] = []
    # store data in sequence/label_seq+'i', whose type is going to be converted from list to ndarray
    for row in rows_1[1:]:
        # store data which come from one sequence
        if row[0] == str(veh_num):
            globals()[seq].append(row[1:])
            # length(array: n_sequence,) counts the time steps in one sequence
            length[veh_num-1] = length[veh_num-1] + 1
        # detect changes on sequence (veh_id), initialize a new array and then write data
        else:
            globals()[seq] = np.array(globals()[seq], dtype=float)
            seq = 'sequence_' + row[0]
            veh_num = veh_num + 1
            globals()[seq] = []
            globals()[seq].append(row[1:])
            length[veh_num-1] = length[veh_num-1] + 1
    # convert the type of sequence+'i' from list to array
    globals()[seq] = np.array(globals()[seq], dtype=float)

# extract features
    # features(array: number of total sequences*number of categories)
    features = np.empty((np.sum(length), 6))
    # seq_range(array: n_sequence+1,): each element represent the start row of each sequence in the feature vector
    seq_range = np.zeros((n_sequence + 1,), dtype=int)
    for i in range(1, np.size(length, 0)+1):
        seq = 'sequence_' + str(i)
        features[seq_range[i-1]:seq_range[i-1]+length[i-1]] = globals()[seq]
        seq_range[i] = np.array([np.sum(length[:i])])

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

# set saturation values for heading angle and yaw rate
    if satruation == 1:
        for i in range(np.size(features,0)):
            if features[i,4] > 2.5:
                features[i,4] = 2.5
            elif features[i,4] < -2.5:
                features[i,4] = -2.5
            if features[i,5] > 0.8:
                features[i,5] = 0.8
            elif features[i,5] < -0.8:
                features[i, 5] = -0.8

# delete useless features
    if n_feature == 4:
        features = np.delete(features, [1, 3], 1)
    elif n_feature == 3:
        features = np.delete(features, [0, 1, 3], 1)

# return value
    return features, label, seq_range


def feat_clip(features, label, seq_range):

    """
    funtion feat_clip (features, label, seq_range)

    :param
        features(array: n_seq * n_feature): result from readcsvdata.py/concatenate_features.py
        label(array: n_seq,): result from readcsvdata.py/concatenate_features.py
        seq_range(array: n_sequence+1, ): result from readcsvdata.py/concatenate_features.py
    :return:
        features_new(array: n_seq_new * n_feature): 20s before the start point and 30s after the start point
        label_new(array: n_seq_new,): 20s before the start point and 30s after the start point
        seq_range_new(array: n_sequence+1,): 20s before the start point and 30s after the start point
    """

# how many sequences inside this data set
    n_sequence = np.size(seq_range, 0) - 1

# initialize variables
    start_point = np.zeros((n_sequence,), dtype=int)
    seq_range_new = np.zeros((n_sequence+1,), dtype=int)
    features_new = np.zeros((20 * n_sequence, np.size(features, 1)))
    label_new = np.zeros((20 * n_sequence,), dtype=int)

# find start point, extract part of the features and labels, update new seq_range
    for i in range(n_sequence):
        start = seq_range[i]
        stop = seq_range[i + 1]
        # find the start point by labels
        start_point[i] = np.argwhere(label[start:stop] == 1)[0] + seq_range[i]
        # extract features and labels (2 sec before and 3 sec after the maneuver begins)
        if start_point[i]-10 > seq_range[i]:
            # from 2 sec before to 3 sec after the maneuver begins
            seq_range_new[i+1] = seq_range_new[i] + 20
            features_new[seq_range_new[i]:seq_range_new[i+1]] = features[start_point[i]-10:start_point[i]+10]
            label_new[seq_range_new[i]:seq_range_new[i+1]] = label[start_point[i]-10:start_point[i]+10]
        else:
            # from the start of the sequence to 3 sec after the maneuver begins
            seq_range_new[i+1] = seq_range_new[i] + start_point[i]+20 - seq_range[i]
            features_new[seq_range_new[i]:seq_range_new[i+1]] = features[seq_range[i]:start_point[i]+10]
            label_new[seq_range_new[i]:seq_range_new[i+1]] = label[seq_range[i]:start_point[i]+10]
    # delete unused rows from features_new and label_new
    features_new = np.delete(features_new, np.s_[seq_range_new[n_sequence]:], 0)
    label_new = np.delete(label_new, np.s_[seq_range_new[n_sequence]:], 0)

# return value
    return features_new, label_new, seq_range_new
