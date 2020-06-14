import Constants
from Data import create_dataset, my_pdb_parser
from GNN.GNNModel import GNNModel
from PlotMPL import plot_from_file

dataset, dataset_filenames = create_dataset(Constants.PDB_PATH, limit=10)

my_model = GNNModel()
my_model.train(dataset)


my_pdb_parser(dataset_filenames[0], do_plot=True)
