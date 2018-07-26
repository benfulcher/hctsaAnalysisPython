'''
Created on 8 Jun 2015
@author: philip knaute
------------------------------------------------------------------------------
Copyright (C) 2015, Philip Knaute <philiphorst.project@gmail.com>,
This work is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of
this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send
a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
California, 94041, USA.
------------------------------------------------------------------------------
'''
import scipy.io as sio

def read_calc_times(mat_file_path):
    """ Read the average calculation times for each operation from a HCTSA_loc.mat file
    Parameters:
    -----------
    mat_file_path : string
        Path to the HCTSA_loc.mat file
    Returns:
    --------
    op_ids_times_lst : list
        List of two lists. First being the operation ids and second the average times over all
        calculated timeseries for each respective operation.
    """
    mat_file = sio.loadmat(mat_file_path)
    op_id = lambda i : int(mat_file['Operations'][i][0][3][0])

    calc_times = mat_file['TS_CalcTime'].sum(axis=0)/mat_file['TS_CalcTime'].shape[0]
    op_ids = [op_id(i) for i in range(mat_file['Operations'].shape[0])]
    return [op_ids,calc_times.tolist()]

def read_from_mat_file(mat_file_path,hctsa_struct_names,is_from_old_matlab = False):
    """
    read .mat files into appropriate python data structures

    Parameters:
    ----------
    mat_file_path : string
        Path to the matlab file to be read
    hctsa_struct_names : list
        List of strings of identifiers for which structures are to be read from the mat_file.
        Possible values are : 'TimeSeries','Operations','TS_DataMat'
    is_from_old_matlab : bool
        If the HCTSA_loc.mat files are saved from an older version of the comp engine. The order of entries is different.
    Returns:
    --------
    retval : tuple
        Tuple of the imported values in the order given by hctsa_struct_names
    """
    mat_file = sio.loadmat(mat_file_path)
    retval = tuple()
    for item in hctsa_struct_names:
        if item == 'TimeSeries':
            timeseries = dict()
            if is_from_old_matlab:
                # lambda function used to populate the dictionary with the appropriate data lists
                ts_id = lambda i : int(mat_file['TimeSeries'][i][0][0][0])
                ts_filename = lambda i : str(mat_file['TimeSeries'][i][0][1][0])
                ts_kw = lambda i : str(mat_file['TimeSeries'][i][0][2][0])
                ts_n_samples = lambda i : int(mat_file['TimeSeries'][i][0][3][0])
                # data is not included in the returned dictionary as it seem a waste of space
                #ts_data = lambda i : mat_file['TimeSeries'][i][0][4]
                for extractor,key in zip([ts_id,ts_filename ,ts_kw,ts_n_samples],['id','filename','keywords','n_samples']):
                    timeseries[key] =[extractor(i) for i in range(mat_file['TimeSeries'].shape[0])]
            else:
                # -- currently there seems to be a bug in the creation of those files. Need to
                #    read them differently
                # lambda function used to populate the dictionary with the appropriate data lists
                #ts_id = lambda i : int(mat_file['TimeSeries'][i][0][0][0])

                # ----------- this is the original version, I (Carl) somehow need to switch the first two dimensions ----
                # ts_filename = lambda i : str(mat_file['TimeSeries'][0][i][0][0])
                # ts_kw = lambda i : str(mat_file['TimeSeries'][0][i][1][0])
                # ts_n_samples = lambda i : int(mat_file['TimeSeries'][0][i][2][0])
                # # -- data is not included in the returned dictionary as it seem a waste of space
                # #ts_data = lambda i : mat_file['TimeSeries'][i][0][3]
                # for extractor,key in zip([ts_filename ,ts_kw,ts_n_samples],['filename','keywords','n_samples']):
                #     timeseries[key] =[extractor(i) for i in range(mat_file['TimeSeries'].shape[1])]

                # ------------ to this
                ts_filename = lambda i: str(mat_file['TimeSeries'][i][0][0][0])
                ts_kw = lambda i: str(mat_file['TimeSeries'][i][0][1][0])
                ts_n_samples = lambda i: int(mat_file['TimeSeries'][i][0][2][0])
                # -- data is not included in the returned dictionary as it seem a waste of space
                # ts_data = lambda i : mat_file['TimeSeries'][i][0][3]
                for extractor, key in zip([ts_filename, ts_kw, ts_n_samples], ['filename', 'keywords', 'n_samples']):
                    timeseries[key] = [extractor(i) for i in range(mat_file['TimeSeries'].shape[0])]

            retval = retval + (timeseries,)

        if item == 'Operations':
            operations = dict()
            if is_from_old_matlab:
                op_id = lambda i : int(mat_file['Operations'][i][0][0][0])
                op_name = lambda i : str(mat_file['Operations'][i][0][1][0])
                op_kw = lambda i : str(mat_file['Operations'][i][0][2][0])
                op_code = lambda i : str(mat_file['Operations'][i][0][3][0])
                op_mopid = lambda i : int(mat_file['Operations'][i][0][4][0])
            else:
                # lambda function used to populate the dictionary with the appropriate data lists
                op_id = lambda i : int(mat_file['Operations'][i][0][3][0])
                op_name = lambda i : str(mat_file['Operations'][i][0][1][0])
                op_kw = lambda i : str(mat_file['Operations'][i][0][2][0])
                op_code = lambda i : str(mat_file['Operations'][i][0][0][0])
                op_mopid = lambda i : int(mat_file['Operations'][i][0][4][0])


            for extractor,key in zip([op_id,op_name ,op_kw,op_code,op_mopid],['id','name','keywords','code_string','master_id']):
                operations[key] =[extractor(i) for i in range(mat_file['Operations'].shape[0])]
            retval = retval + (operations,)

        if item == 'TS_DataMat':
            retval = retval+(mat_file['TS_DataMat'],)

        if item == 'MasterOperations':
            m_operations = dict()
            if is_from_old_matlab:
                raise NameError('Don''t know how to get MasterOperations from old Matlab version.')
            else:
                # lambda function used to populate the dictionary with the appropriate data lists
                m_op_id = lambda i: int(mat_file['MasterOperations'][i][0][2][0][0])
                m_op_name = lambda i: str(mat_file['MasterOperations'][i][0][1][0])

            for extractor, key in zip([m_op_id, m_op_name],
                                      ['id', 'name']):
                m_operations[key] = [extractor(i) for i in range(mat_file['MasterOperations'].shape[0])]
            retval = retval + (m_operations,)

return retval
