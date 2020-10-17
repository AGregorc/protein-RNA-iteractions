"""
This example client takes a PDB file, sends it to the REST service, which
creates dSSP data. The dSSP data is then output to the console.

"""

import argparse
import json
import os

import requests
import time

REST_URL = 'https://www3.cmbi.umcn.nl/xssp/'
DSSP_DIR = 'dssp/'
PDB_DIR = 'pdbs/'


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
    print("Job submitted successfully. Id is: '{}'".format(job_id))

    # Loop until the job running on the server has finished, either successfully
    # or due to an error.
    ready = False
    while not ready:
        # Check the status of the running job. If an error occurs an exception
        # is raised and the program exits. If the request is successful, the
        # status is returned.
        url_status = '{}api/status/pdb_file/dssp/{}/'.format(rest_url,
                                                                  job_id)
        r = requests.get(url_status)
        r.raise_for_status()

        status = json.loads(r.text)['status']
        print("Job status is: '{}'".format(status))

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
            raise Exception(json.loads(r.text)['message'])
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


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(DSSP_DIR):
        os.makedirs(DSSP_DIR)

    pdbs = []
    with open('pdbs.lst') as f:
        for pdb in f:
            pdbs.append(pdb.strip())

    for pdb in pdbs:
        filename = f'{pdb}.dssp'
        if not os.path.exists(os.path.join(DSSP_DIR, filename)):
            result = pdb_to_dssp(os.path.join(PDB_DIR, pdb + '.pdb'), REST_URL)
            open(DSSP_DIR + filename, 'w').write(result)
