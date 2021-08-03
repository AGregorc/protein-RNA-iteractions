import json
import os
import time
import traceback

import requests
from tqdm import tqdm

import Constants
from Data.utils import get_analysis_pdb_list

URL_RCSB = 'https://files.rcsb.org/view/'
DSSP_REST_URL = 'https://www3.cmbi.umcn.nl/xssp/'
QUERY_ROWS = 10000

ALL_PROTEINS_QUERY = 'https://search.rcsb.org/rcsbsearch/v1/query?json=%7B%22query%22%3A%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C' \
                     '%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22' \
                     '%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22greater%22%2C%22negation' \
                     '%22%3Afalse%2C%22value%22%3A0%2C%22attribute%22%3A%22rcsb_entry_info.polymer_entity_count_protein%22%7D%7D%2C%7B' \
                     '%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22less%22%2C' \
                     '%22negation%22%3Afalse%2C%22value%22%3A62%2C%22attribute%22%3A%22rcsb_entry_info' \
                     '.deposited_polymer_entity_instance_count%22%7D%7D%5D%7D%2C%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A' \
                     '%22or%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B' \
                     '%22operator%22%3A%22range%22%2C%22negation%22%3Afalse%2C%22value%22%3A%5B1%2C1.5%5D%2C%22attribute%22%3A' \
                     '%22rcsb_entry_info.resolution_combined%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C' \
                     '%22parameters%22%3A%7B%22operator%22%3A%22range%22%2C%22negation%22%3Afalse%2C%22value%22%3A%5B1.5%2C2%5D%2C' \
                     '%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22' \
                     '%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22range%22%2C%22negation%22%3Afalse%2C%22value%22%3A%5B2%2C2' \
                     '.5%5D%2C%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C' \
                     '%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22range%22%2C%22negation%22%3Afalse%2C%22value' \
                     '%22%3A%5B2.5%2C3%5D%2C%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%7D%7D%2C%7B%22type%22%3A' \
                     '%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22range%22%2C%22negation%22' \
                     '%3Afalse%2C%22value%22%3A%5B3%2C3.5%5D%2C%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%7D%7D%5D%7D%5D' \
                     '%7D%2C%22return_type%22%3A%22entry%22%2C%22request_options%22%3A%7B%22pager%22%3A%7B%22start%22%3A{}%2C%22rows%22' \
                     '%3A' + str(QUERY_ROWS) + '%7D%7D%7D'

RNA_PROTEIN_QUERY = 'https://search.rcsb.org/rcsbsearch/v1/query?json=%7B%22query%22%3A%7B%22type%22%3A%22group%22%2C' \
                    '%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C' \
                    '%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22terminal%22%2C' \
                    '%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22greater%22%2C%22negation' \
                    '%22%3Afalse%2C%22value%22%3A0%2C%22attribute%22%3A%22rcsb_entry_info.polymer_entity_count_RNA%22' \
                    '%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B' \
                    '%22operator%22%3A%22greater%22%2C%22negation%22%3Afalse%2C%22value%22%3A0%2C%22attribute%22%3A' \
                    '%22rcsb_entry_info.polymer_entity_count_protein%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C' \
                    '%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22less%22%2C%22negation%22' \
                    '%3Afalse%2C%22value%22%3A62%2C%22attribute%22%3A%22rcsb_entry_info' \
                    '.deposited_polymer_entity_instance_count%22%7D%7D%5D%7D%2C%7B%22type%22%3A%22group%22%2C' \
                    '%22logical_operator%22%3A%22or%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22terminal%22%2C%22service' \
                    '%22%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22range%22%2C%22negation%22%3Afalse' \
                    '%2C%22value%22%3A%5B1%2C1.5%5D%2C%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%7D' \
                    '%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B' \
                    '%22operator%22%3A%22range%22%2C%22negation%22%3Afalse%2C%22value%22%3A%5B1.5%2C2%5D%2C' \
                    '%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%7D%7D%2C%7B%22type%22%3A%22terminal' \
                    '%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22range%22%2C' \
                    '%22negation%22%3Afalse%2C%22value%22%3A%5B2%2C2.5%5D%2C%22attribute%22%3A%22rcsb_entry_info' \
                    '.resolution_combined%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C' \
                    '%22parameters%22%3A%7B%22operator%22%3A%22range%22%2C%22negation%22%3Afalse%2C%22value%22%3A%5B2' \
                    '.5%2C3%5D%2C%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%7D%7D%2C%7B%22type%22%3A' \
                    '%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22operator%22%3A%22range%22' \
                    '%2C%22negation%22%3Afalse%2C%22value%22%3A%5B3%2C3.5%5D%2C%22attribute%22%3A%22rcsb_entry_info' \
                    '.resolution_combined%22%7D%7D%5D%7D%5D%7D%2C%22return_type%22%3A%22entry%22%2C%22request_options' \
                    '%22%3A%7B%22pager%22%3A%7B%22start%22%3A{}%2C%22rows%22%3A' + str(QUERY_ROWS) + '%7D%7D%7D '

PDB_DIR = Constants.PDB_PATH
DSSP_DIR = Constants.DSSP_PATH


def pdb_to_dssp(path, filename, rest_url):
    # Read the pdb file data into a variable
    files = {'file_': Constants.open_data_file(path, filename).read()}

    # Send a request to the server to create dssp data from the pdb file data.
    # If an error occurs, an exception is raised and the program exits. If the
    # request is successful, the id of the job running on the server is
    # returned.
    url_create = '{}api/create/pdb_file/dssp/'.format(rest_url)
    r = requests.post(url_create, files=files, timeout=20)

    job_id = json.loads(r.text)['id']
    # print("Job submitted successfully. Id is: '{}'".format(job_id))

    # Loop until the job running on the server has finished, either successfully
    # or due to an error.
    ready = False
    while not ready:
        # Check the status of the running job. If an error occurs an exception
        # is raised and the program exits. If the request is successful, the
        # status is returned.
        url_status = '{}api/status/pdb_file/dssp/{}/'.format(rest_url, job_id)
        r = requests.get(url_status)
        r.raise_for_status()

        status = json.loads(r.text)['status']
        # print("Job status is: '{}'".format(status))

        # If the status equals SUCCESS, exit out of the loop by changing the
        # condition ready. This causes the code to drop into the `else` block
        # below.
        #
        # If the status equals either FAILURE or REVOKED, an exception is raised
        # containing the error message. The program exits.
        #
        # Otherwise, wait for five seconds and start at the beginning of the
        # loop again.
        if status == 'SUCCESS':
            ready = True
        elif status in ['FAILURE', 'REVOKED']:
            # raise Exception(json.loads(r.text)['message'])
            print(f"Error when loading dssp file {filename}")
            return False
        else:
            time.sleep(1)
    if ready:
        # Requests the result of the job. If an error occurs an exception is
        # raised and the program exits. If the request is successful, the result
        # is returned.
        url_result = '{}api/result/pdb_file/dssp/{}/'.format(rest_url, job_id)
        r = requests.get(url_result)
        r.raise_for_status()
        result = json.loads(r.text)['result']

        # Return the result to the caller, which prints it to the screen.
        return result


def load_pdbs_from_list(pdb_list):
    for pdb in tqdm(pdb_list):

        # Load pdb file
        pdb_filename = Constants.to_pdb_filename(pdb)
        if not Constants.data_file_exists(PDB_DIR, pdb_filename):
            try:
                r = requests.get(URL_RCSB + pdb_filename, allow_redirects=True, timeout=8)
                # r.raise_for_status()
                if r.status_code < 300:
                    with Constants.open_data_file(PDB_DIR, pdb_filename, read=False) as f:
                        f.write(r.content)
                else:
                    print(f"Error when loading pdb file {pdb} status {r.status_code}")
                    continue
                # print(f'{pdb_filename} added')
            except:
                continue

        # Load dssp file
        dssp_filename = Constants.to_dssp_filename(pdb)
        if not Constants.data_file_exists(DSSP_DIR, dssp_filename):
            result = pdb_to_dssp(PDB_DIR, pdb_filename, DSSP_REST_URL)
            if result is not False:
                with Constants.open_data_file(DSSP_DIR, dssp_filename, read=False) as f:
                    f.write(result)
            # print(f'{dssp_filename} added')


def load_preprocessed_data(pdb_list):
    url = Constants.DATA_API_URL + '/api/preprocessed_file/'
    for pdb in tqdm(pdb_list):

        # Load preprocessed file
        pdb_filename = f'{pdb}{Constants.GRAPH_EXTENSION}'
        path = os.path.join(Constants.SAVED_GRAPH_PATH, pdb_filename)
        if not os.path.exists(path):
            try:
                r = requests.get(url + pdb_filename, timeout=8)
                # r.raise_for_status()
                if r.status_code < 300:
                    with open(path, 'wb') as f:
                        f.write(r.content)
                else:
                    print(f"Error when loading pdb file {pdb} status {r.status_code}")
                    continue
                # print(f'{pdb_filename} added')
            except:
                continue


def load_data_for_analysis(start_pdb=0, limit=None):
    Constants.makedir_if_not_exists(PDB_DIR)
    Constants.makedir_if_not_exists(DSSP_DIR)

    all_pdbs = get_analysis_pdb_list(start_pdb, limit)

    load_pdbs_from_list(all_pdbs)


def get_pdb_list(query, limit=None, ignore=None):
    if ignore is None:
        ignore = []
    new_pdbs = []
    start_row = 0

    while True:
        print(f'Loading pdb ids from start row {start_row}')
        url_result = query.format(start_row)
        r = requests.get(url_result)
        r.raise_for_status()
        result = json.loads(r.text)['result_set']
        for elem in result:
            curr_pdb = elem['identifier'].lower()
            if curr_pdb not in ignore:
                new_pdbs.append(curr_pdb)
        start_row += QUERY_ROWS
        if len(result) < QUERY_ROWS or (limit is not None and start_row >= limit):
            break

    if limit is not None:
        new_pdbs = new_pdbs[:limit]
    return new_pdbs


def update_pdbs_list_and_load(query, limit=None, filename='all_pdbs.lst', load_pdbs=True):
    pdbs = []
    with open(os.path.join(Constants.DATA_PATH, filename)) as f:
        for pdb in f:
            pdbs.append(pdb.strip())

    try:
        new_pdbs = get_pdb_list(query, limit, ignore=pdbs)

        if load_pdbs:
            load_pdbs_from_list(new_pdbs)
    except Exception as e:
        print(f'Error on update_pdbs_list_and_load: {e}')
        traceback.print_exc()
        return False
    else:
        # If loading is successful then update pdb list file
        print(len(new_pdbs), len(pdbs))

        with open(os.path.join(Constants.DATA_PATH, filename), 'w') as f:
            first = True
            for pdb in pdbs:
                if first:
                    f.write(pdb)
                    first = False
                    continue
                f.write('\n' + pdb)
            for pdb in new_pdbs:
                f.write('\n' + pdb)

    return True

#
# if __name__ == '__main__':
#     update_pdbs_list()
#     # limit = int(input('How many pdbs do you want to load? '))
#     # load_data(limit=limit)
