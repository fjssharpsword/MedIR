#!/usr/bin/env python

# file: $(NEDC_NFC)/src/classes/search/demo_search_helpers.py
#
# This file contains some useful Python functions and classes that are used
# in the nedc scripts.
#
#------------------------------------------------------------------------------
import re
import os
import ntpath
import csv

BASE_DIR = "/Users/elliottkrome/nedc/cohort_search/"

KEYPHRASE_DICT = {"generalized shaking": "files_01",
                  "TIRDA": "files_04",
                  "status epilepticus": "files_06",
                  "sharp and slow waves": "files_07",
                  "sharp waves": "files_09",
                  "generalized spike and wave": "files_10"}

# function: get_cohort_dir
#
# arguments: in_string_a: a key phrase to look for in KEYPHRASE_DICT
#
# returns: phrase_dir: a dir corresponding to one cohort
#
# this function will return a cohort directory by looking up a phrase
# in KEYPHRASE_DICT. Will return None if no hit is found. This is a cheap demo.
#
def get_cohort_dir(in_string_a):

    # iterate over the keyphrase dictionary and search for matches
    #
    for keyphrase in KEYPHRASE_DICT:

        # if we have a hit, then return directory of edf files
        # associated with hit
        #
        if re.search(keyphrase, in_string_a, re.IGNORECASE):
            phrase_dir = BASE_DIR + KEYPHRASE_DICT[keyphrase]
            return phrase_dir

# function: get_results
#
# arguments:
#  -base_dir_a:      dir corresponding to one cohort
#  -checkbox_dict_a: dict of label types
#
# returns: 
#  -results:    a dict of results, where each results is a dict with
#               edf_path, txt_path, and an edf_name
#  -similarity: a similarity.csv file
#
# this function accumulates all information from a one cohort directory
#
def get_results(base_dir_a,
                checkbox_dict_a):

    # initialize a dictionary which will store all results for one query
    #
    results = {}

    # get similarity_matrix of similarities
    #
    similarity, data_indices = _get_csv_data(base_dir_a)

    # walk the directory file listed and generate full path to each entry
    #
    for dir_entry in os.listdir(base_dir_a):
        dir_entry = base_dir_a + os.sep + dir_entry

        # we are looking only for directories.
        #
        if os.path.isdir(dir_entry):

            # each directory at this level stores the stuff for one search entry
            # initialize a dict for this one entry, and gather information
            # from the directory by iterating over the files
            #
            one_result = {}
            for file_path in os.listdir(dir_entry):

                # store path to edf file
                #
                if file_path.endswith('.edf'):
                    one_result['edf_file'] = dir_entry + os.sep + file_path

                # get path to txt file, also, the text file name tends to
                # include a string that will be good for naming the
                # entry, so collect the name (but replace txt with edf)
                #
                if file_path.endswith('.txt'):

                    # store path to txt file
                    #
                    one_result['txt'] = dir_entry + os.sep + file_path

                    # store name for entry
                    #
                    one_result['edf_name'] = ntpath.basename(file_path).\
                                             replace('txt', 'edf')

            # the name of the report file seems to the only thing
            # to which we can map the ordering data_index.csv.
            #
            index = _map_one_result_to_similarity_index(data_indices,
                                                        one_result['edf_name'])

            # store one result
            #
            results[index] = one_result

    # return dictionary of results and path to similarity csv file
    #
    return (results, similarity)

# function: sort_results
#
# arguments:
#  -results_a:    a dict of results, where each results is a dict with
#                 edf_path, txt_path, and an edf_name
#  -similarity_a: the path to a similarity.csv file
#  -index_a:      index corresponding to DemoSearch.current_file's position in
#                 similarity_a. None if file not accounted for in similarity_a
#
# returns:
#  -sorted_results: dict of results, with keys reordered to reflect similarity
#                   to DemoSearch.current_file
#
# this function sorts a set of results to reflect similarity data
#
def sort_results(results_a,
                 similarity_a,
                 index_a):

    # initialize a dictionary to store sorted results
    #
    sorted_results = {}

    # if DemoSearch.current_file not in results, act as if it is the first
    # entry. (We have to sort according to something)
    #
    if index_a is not None:
        row_number = index_a
    else:
        row_number = 0

    # open the similarity.csv file and parse it as similarity_matrix
    #
    with open(similarity_a) as similarity_file:
        similarity_matrix = csv.reader(similarity_file)

        # loop over rows in similarity_matrix until we find the row corresponding
        # to DemoSearch.current_file
        #
        for row_counter, row in enumerate(similarity_matrix):
            if row_counter == row_number:

                # having found the correct row, accumulate the data
                # column by column, storing in entry_list
                #
                entry_list = []
                for col_counter in (range(len(row))):
                    entry_list.append((col_counter, float(row[col_counter])))

    # define a function for use by `sorted'
    #
    def _get_second_index(item):
        return item[1]

    # sort by second entry in entry_list (similarity)
    #
    sorted_indices = sorted(entry_list,
                            key=_get_second_index,
                            reverse=True)

    # now that we have sorted indices, rebuild the dictionary
    # results_a to correspond the sorting
    #
    for index in range(len(sorted_indices)):
        index_of_next_entry = sorted_indices[index][0]
        try:
            sorted_results[index] = results_a[index_of_next_entry]
        except:
            pass

    return sorted_results

def _map_one_result_to_similarity_index(data_indices_a,
                                        report_name_a):
    counter = 0
    for index in data_indices_a:
        if re.search(index, report_name_a):
            return counter
        else:
            counter += 1

def _get_csv_data(base_dir_a):
    for dir_entry in os.listdir(base_dir_a):
        dir_entry = base_dir_a + os.sep + dir_entry
        if dir_entry.endswith('data_index.csv'):
            data_indices = _get_data_indices(dir_entry)
        elif dir_entry.endswith('similarity.csv'):
            similarity = dir_entry
        else:
            pass
    return similarity, data_indices

def _get_data_indices(dir_path_a):
    data_indices = []
    with open(dir_path_a) as data_index_file:
        reader = csv.reader(data_index_file)
        for row in reader:
            data_indices.append(row[0])

    return data_indices

# NOT WORKING, PLACEHOLDER
#
def _filter_for_annotation_type(edf_file_a,
                               checkbox_dict_a):
    annotations_file = edf_file_a.replace('edf', 'lbl')
    if os.path.isfile(annotations_file):
        has_all_req_anno_types = _check_for_presence_of_annotations(
            edf_file_a,
            annotations_file,
            checkbox_dict_a)
        if has_all_req_anno_types is True:
            return True
        else:
            return False
    else:
        return True

# NOT WORKING, PLACEHOLDER
#
def _check_for_presence_of_annotations(edf_file_a,
                                       annotations_file_a,
                                       checkbox_dict_a):
    print ("TODO: demo_search_helpers._check_for_presence_of_annotations: ")
    print ("implement using demo_annotator")
