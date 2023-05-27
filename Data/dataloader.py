import os
import numpy as np
import scipy.io as sio


def ROI_loader(subject, fil):

    data_path = str(os.path.dirname(os.path.abspath(__file__))) + '/data/subjects'
    all_data = sio.loadmat(data_path + subject + '/' + fil)      
    ROI = all_data['meta']
    Gordon_areas = ROI[0][0][11][0][14]   
    try:
        data = all_data['examples']
    except KeyError:
        data = all_data['examples_passagesentences']
    return Gordon_areas, data


def coltocoord_ROI_ordering(subject, fil):
    
    data_path = str(os.path.dirname(os.path.abspath(__file__))) + '/data/subjects'
    all_data = sio.loadmat(data_path + subject + '/' + fil)      
    ROI = all_data['meta']
    coord = ROI[0][0][5]
    return coord

def matcher(area, last_dim, last, Roi_coord):
    difference = np.sum(last_dim[1]) - np.sum(last[1])
    assert 0<=difference
    assert np.sum(last[1]) == area.shape[1]
    if last_dim.shape[1]>last.shape[1]:
        helper = np.zeros_like(last_dim)
        helper[:,:last.shape[1]] = last
        last = helper
        checker = 0
    else:
        checker = last.shape[1] - last_dim.shape[1]
        helper = np.zeros_like(last)
        helper[:,checker:] = last_dim
        last_dim = helper

    area_new = np.zeros((area.shape[0], np.sum(last_dim[1])))
    counter = holder = index = index_old = 0
    mean = np.zeros((area.shape[0],1))
    while counter<last_dim.shape[1]:
        if last_dim[1,counter] > (last[1,counter]+holder) and checker == 0:
            if last[1,counter] == 0:
                if last[1,counter-1] != 0:
                    mean = np.reshape(np.mean(area[:,(Roi_coord[:,2] == last[0,counter-1])], axis=1),(area.shape[0],1))
                area_new[:,index+last[1,counter]:index+last_dim[1,counter]-holder] = np.tile(mean,(1,last_dim[1,counter] - (last[1,counter]+holder)))
                index+= (last_dim[1,counter]-holder)
            else:
                mean = np.reshape(np.mean(area[:,(Roi_coord[:,2] == last[0,counter])], axis=1),(area.shape[0],1))
                area_new[:,index:index+last[1,counter]] = area[:,index_old:index_old+last[1,counter]]
                if difference > 0:
                    update = last_dim[1,counter] - (last[1,counter]+holder)
                    if (difference - update) >= 0:
                        area_new[:,index+last[1,counter]:index+last_dim[1,counter]-holder] = np.tile(mean,(1,update))
                        index += (last_dim[1,counter] - holder)
                        difference = difference - update
                    else:
                        area_new[:,index+last[1,counter]:index+last[1,counter]+difference] = np.tile(mean,(1,difference))
                        index += (difference + last[1,counter])
                        difference = 0
                else:
                    index+= last[1,counter]
            holder = 0
        else:
            area_new[:,index:index+last[1,counter]] = area[:,index_old:index_old+last[1,counter]]
            index += last[1,counter]
            if checker > 0:
                checker-=1
                holder += last[1,counter]
            else:
                holder = (last[1,counter] + holder) - last_dim[1,counter]
        index_old+=last[1,counter]

        counter+=1
    # assert holder == 0
    return area_new

def class_sizer(subject):
    no_sent = ['M05','M06','M10', 'M13','M16', 'M17']
    only_two = ['M03']
    only_three = [ 'M08', 'M09','M14']
    both = ['P01','M02','M03','M04','M07','M15']
    values = [4530, 4287, 4146, 3903]
    values_test = [0, 243, 384, 627]
    if subject in no_sent:
        return values[0], values_test[0]
    if subject in only_two:
        return values[1], values_test[1]
    if subject in only_three:
        return values[2], values_test[2]
    if subject in both:
        return values[3], values_test[3]

def dataloader_sentence_word_split_new_matching_all_subjects(subject):

    data_path = str(os.path.dirname(os.path.abspath(__file__))) + '/data/subjects/'
    vector_path_180 = str(os.path.dirname(os.path.abspath(__file__))) + '/data/glove_data/180_concepts_real.mat'
    vector_path_243 = str(os.path.dirname(os.path.abspath(__file__))) + '/data/glove_data/243_sentences_real.mat'
    vector_path_384 =str(os.path.dirname(os.path.abspath(__file__))) +  '/data/glove_data/384_sentences_real.mat'

    vector_180 = sio.loadmat(vector_path_180)['data']
    vector_243 = sio.loadmat(vector_path_243)['data']
    vector_384 = sio.loadmat(vector_path_384)['data']

    sizes = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/sizes.npy')
    last_dim_all = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/last_dim.npz')

    subjects = ['P01','M02','M03','M04','M05','M06','M07', 'M08', 'M09','M15','M10', 'M13','M14','M16', 'M17']  #,'M10', 'M13','M14','M16','M17'
    value, value_test = class_sizer(subject)

    data_train = np.zeros((value,65730)) 
    data_fine = np.zeros((7560,65730)) 
    data_test = np.zeros((value_test,65730))  
    data_fine_test = np.zeros((540,65730))

    glove_train = np.zeros((value,300))   
    glove_fine = np.zeros((7560,300))   
    glove_test = np.zeros((value_test,300))  
    glove_fine_test = np.zeros((540,300))

    numb = numb_test = numb_fine_test = numb_fine = 0
    tot = tot_fine = tot_test = tot_fine_test = 0

    for sub in subjects:

        folder = os.listdir(data_path + sub)
        values = np.zeros((627,212742))     
        values_fine = np.zeros((540,212742))
        numb_fine_tes = numb_fine1 = numb_tes = numb1 = 0
        for fil in folder:
            Gordon, data = ROI_loader(sub,fil)
            coord = coltocoord_ROI_ordering(sub,fil)
            if sub == subject:
                if fil.startswith('data'):
                    if fil.startswith('data_180'):
                        values_fine[numb_fine_tes:numb_fine_tes+data.shape[0],:data.shape[1]] = data
                        if data.shape[0]==180:
                            glove_fine_test[numb_fine_test:numb_fine_test+data.shape[0],:] = vector_180
                        numb_fine_test +=data.shape[0]
                        numb_fine_tes += data.shape[0]
                    else:
                        values[numb_tes:numb_tes+data.shape[0],:data.shape[1]] = data
                        if data.shape[0]==243:
                            glove_test[numb_test:numb_test+data.shape[0],:] = vector_243
                        if data.shape[0]==384:
                            glove_test[numb_test:numb_test+data.shape[0],:] = vector_384
                        numb_test +=data.shape[0]
                        numb_tes += data.shape[0]
            else:
                if fil.startswith('data'):
                    if fil.startswith('data_180'):
                        values_fine[numb_fine1:numb_fine1+data.shape[0],:data.shape[1]] = data
                        if data.shape[0]==180:
                            glove_fine[numb_fine:numb_fine+data.shape[0],:] = vector_180
                        numb_fine +=data.shape[0]
                        numb_fine1+=data.shape[0]
                    else:
                        values[numb1:numb1+data.shape[0],:data.shape[1]] = data
                        if data.shape[0]==243:
                            glove_train[numb:numb+data.shape[0],:] = vector_243
                        if data.shape[0]==384:
                            glove_train[numb:numb+data.shape[0],:] = vector_384
                        numb +=data.shape[0]
                        numb1+=data.shape[0]
        values = values[~(values==0).all(1)]
        ind_array = 0
        for i in range(333):
            last_dim = last_dim_all['arr_' + str(i)]
            Roi_coord = np.squeeze(coord[Gordon[i][0]])
            last = np.asarray(np.unique(Roi_coord[:,2], return_counts=True))
            big = np.sum(last_dim[1])
            assert big == sizes[i]
            indexes = Gordon[i][0]
            area = values[:,indexes]
            area_fine = values_fine[:,indexes]
            if area.shape[0]!=0:
                area = np.reshape(area, (values.shape[0],-1))
                area = matcher(area, last_dim, last, Roi_coord)
            area_fine = np.reshape(area_fine, (values_fine.shape[0],-1))
            area_fine = matcher(area_fine, last_dim, last, Roi_coord)

            if sub == subject:
                if area.shape[0]!=0:
                    data_test[tot_test:tot_test+values.shape[0],ind_array:ind_array+area.shape[1]] = area
                data_fine_test[tot_fine_test:tot_fine_test+values_fine.shape[0],ind_array:ind_array+area_fine.shape[1]] = area_fine
            else :
                if area.shape[0]!=0:
                    data_train[tot:(tot+values.shape[0]),ind_array:ind_array+area.shape[1]] = area
                data_fine[tot_fine:(tot_fine+values_fine.shape[0]),ind_array:ind_array+area_fine.shape[1]] = area_fine
            ind_array+=sizes[i]
        if sub == subject:
            tot_test+=values.shape[0]
            tot_fine_test+= values_fine.shape[0]
        else:
            tot+=values.shape[0]
            tot_fine+= values_fine.shape[0]

    return data_train, data_fine_test, glove_train, glove_fine_test, data_fine, data_fine_test, glove_fine, glove_fine_test
