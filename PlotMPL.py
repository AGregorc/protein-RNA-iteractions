from Bio.PDB import Atom

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from Constants import LABEL_POSITIVE_COLOR, LABEL_NEGATIVE_COLOR
from Data import my_pdb_parser
from Preprocess import get_dgl_id


def use_new_window():
    mpl.use('Qt5Agg')


def plot_graph(pairs=None, atoms=None, atom_color_func=lambda atom: None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if title is not None:
        ax.set_title(title)  # , loc='left')
    if pairs:
        for pair in pairs:
            a1, a2 = pair
            if not isinstance(a1, Atom.Atom) and hasattr(a1, 'get_atoms'):
                g1 = a1.get_atoms()
                g2 = a2.get_atoms()
                a1 = next(g1)
                a2 = next(g2)
            a1v = a1.get_vector()
            a2v = a2.get_vector()
            x = [a1v[0], a2v[0]]
            y = [a1v[1], a2v[1]]
            z = [a1v[2], a2v[2]]
            ax.plot(x, y, z, color='grey', linewidth=1)

    if atoms:
        x, y, z = [], [], []
        col = None
        # check if we want colors
        if atom_color_func(atoms[0]):
            col = []

        for atom in atoms:
            vec = atom.get_vector()
            x.append(vec[0])
            y.append(vec[1])
            z.append(vec[2])
            if col is not None:
                col.append(atom_color_func(atom))

        ax.scatter(x, y, z, c=col, s=3, marker='o')
    plt.show()


def plot_from_file(filename, color_func):
    G, atoms, pairs, labels = my_pdb_parser(filename)
    plot_graph(pairs=pairs, atoms=atoms, atom_color_func=color_func, title=filename)


def plot_predicted(filename, model):
    G, atoms, pairs, labels = my_pdb_parser(filename)
    logits = model(G)
    logits = logits.cpu()

    def get_predicted_color(atom):
        dgl_id = get_dgl_id(atom)
        pred = logits[dgl_id].detach().numpy()
        cls = pred.argmax()
        if cls == 1:
            return LABEL_POSITIVE_COLOR
        return LABEL_NEGATIVE_COLOR

    plot_graph(pairs=pairs, atoms=atoms, atom_color_func=get_predicted_color, title=filename)
