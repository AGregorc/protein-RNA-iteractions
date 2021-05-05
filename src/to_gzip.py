import gzip
import os

import Constants

PATHS = [Constants.PDB_PATH, Constants.DSSP_PATH]
GZIP_EXTENSIONS = [".pdb", ".dssp"]


def to_gzip():
    for path, extension in zip(PATHS, GZIP_EXTENSIONS):
        for file in os.listdir(path):
            filepath = os.path.join(path, os.fsdecode(file))
            if filepath.endswith(extension):
                with open(filepath, "r") as f:
                    data = f.read()
                with gzip.open(filepath+".gz", "wt") as fgz:
                    fgz.write(data)
                os.remove(filepath)
            # if filepath.endswith(".gz"):
            #     with gzip.open(filepath, "rt") as fgz:
            #         print(fgz.read())
            #     break


if __name__ == "__main__":
    to_gzip()
