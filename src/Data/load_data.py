import json
import os
import time

import requests
from tqdm import tqdm

import Constants

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
'%3A'+str(QUERY_ROWS)+'%7D%7D%7D'

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
                    '%22%3A%7B%22pager%22%3A%7B%22start%22%3A{}%2C%22rows%22%3A'+str(QUERY_ROWS)+'%7D%7D%7D '


PDB_DIR = Constants.PDB_PATH
DSSP_DIR = Constants.DSSP_PATH


def pdb_to_dssp(pdb_file_path, rest_url):
    # Read the pdb file data into a variable
    files = {'file_': open(pdb_file_path, 'rb')}

    # Send a request to the server to create dssp data from the pdb file data.
    # If an error occurs, an exception is raised and the program exits. If the
    # request is successful, the id of the job running on the server is
    # returned.
    url_create = '{}api/create/pdb_file/dssp/'.format(rest_url)
    r = requests.post(url_create, files=files)
    r.raise_for_status()

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
            print(f"Error when loading dssp file {pdb_file_path}")
            return False
        else:
            time.sleep(1)
    else:
        # Requests the result of the job. If an error occurs an exception is
        # raised and the program exits. If the request is successful, the result
        # is returned.
        url_result = '{}api/result/pdb_file/dssp/{}/'.format(rest_url, job_id)
        r = requests.get(url_result)
        r.raise_for_status()
        result = json.loads(r.text)['result']

        # Return the result to the caller, which prints it to the screen.
        return result


def load_from_list(pdb_list):
    for pdb in tqdm(pdb_list):

        # Load pdb file
        pdb_filename = f'{pdb}.pdb'
        if not os.path.exists(os.path.join(PDB_DIR, pdb_filename)):
            try:
                r = requests.get(URL_RCSB + pdb_filename, allow_redirects=True, timeout=8)
                # r.raise_for_status()
                if r.status_code < 300:
                    with open(os.path.join(PDB_DIR, pdb_filename), 'wb') as f:
                        f.write(r.content)
                else:
                    print(f"Error when loading pdb file {pdb} status {r.status_code}")
                    continue
                # print(f'{pdb_filename} added')
            except:
                continue

        # Load dssp file
        dssp_filename = f'{pdb}.dssp'
        if not os.path.exists(os.path.join(DSSP_DIR, dssp_filename)):
            result = pdb_to_dssp(os.path.join(PDB_DIR, pdb_filename), DSSP_REST_URL)
            if result is not False:
                with open(os.path.join(DSSP_DIR, dssp_filename), 'w') as f:
                    f.write(result)
            # print(f'{dssp_filename} added')


def load_data(start_pdb=0, limit=None):
    if not os.path.exists(PDB_DIR):
        os.makedirs(PDB_DIR)
    if not os.path.exists(DSSP_DIR):
        os.makedirs(DSSP_DIR)

    pdbs = []
    with open(os.path.join(Constants.DATA_PATH, 'pdbs.lst')) as f:
        for pdb in f:
            pdbs.append(pdb.strip())

    print(f'Number of .pdbs: {len(pdbs)}, loading {limit} of them')

    i = 0
    if limit is None:
        all_pdbs = pdbs[start_pdb:]
    else:
        all_pdbs = pdbs[start_pdb:start_pdb + limit]

    load_from_list(all_pdbs)


def update_pdbs_list_and_load(query=ALL_PROTEINS_QUERY):
    pdbs = []
    with open(os.path.join(Constants.DATA_PATH, 'all_pdbs.lst')) as f:
        for pdb in f:
            pdbs.append(pdb.strip())

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
            if curr_pdb not in pdbs:
                new_pdbs.append(curr_pdb)
        start_row += QUERY_ROWS
        if len(result) < QUERY_ROWS:
            break

    load_from_list(new_pdbs)
    print(len(new_pdbs), len(pdbs))

    with open(os.path.join(Constants.DATA_PATH, 'all_pdbs.lst'), 'w') as f:
        first = True
        for pdb in pdbs:
            if first:
                f.write(pdb)
                first = False
                continue
            f.write('\n' + pdb)
        for pdb in new_pdbs:
            f.write('\n' + pdb)

#
# if __name__ == '__main__':
#     update_pdbs_list()
#     # limit = int(input('How many pdbs do you want to load? '))
#     # load_data(limit=limit)
