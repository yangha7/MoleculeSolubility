import numpy as np
from rdkit import Chem

#　Feature Extractor Interface
class Feature:
    def __init__(self):
        return

    def feature(self, mol_string):
        """

        :param mol_string: SMILES descriptor
        :return: numpy feature vector
        """
        return

    def description(self):
        """

        :return: the description for each dimension of features.
        """
        return

class AtomFeature(Feature):
    def __init__(self, atoms):
        super(AtomFeature, self).__init__()
        self.atoms = atoms

    def feature(self, mol_string):
        count = np.zeros(len(self.atoms), np.float32)
        mol = Chem.MolFromSmiles(mol_string)
        for idx, atom_id in enumerate(self.atoms):
            at = Chem.MolFromSmarts('[#{}]'.format(atom_id))
            count[idx] = float(len(mol.GetSubstructMatches(at)))
        print(mol_string, count)
        return count

    def description(self):
        from misc.periodic_table import periodic_table
        return [periodic_table[atom-1] for atom in self.atoms]

class FunctionalGroupFeature(Feature):
    def __init__(self, func_groups=[]):
        super(FunctionalGroupFeature, self).__init__()
        if len(func_groups) > 0:
            self.func_groups = func_groups
        else:
            from misc.functional_groups import FUNCTIONAL_GROUPS_light
            self.func_groups = FUNCTIONAL_GROUPS_light.copy()

    def feature(self, mol_string):
        count = np.zeros(len(self.func_groups), np.float32)
        mol = Chem.MolFromSmiles(mol_string)
        for idx, func_group in enumerate(self.func_groups):
            chem_func_group = Chem.MolFromSmarts(func_group)
            matches = mol.GetSubstructMatches(chem_func_group)
            count[idx] = len(matches)

        return count

    def description(self):
        return self.func_groups

class WeightFeature(Feature):
    def __init__(self):
        super(WeightFeature, self).__init__()

    def feature(self, mol_string):
        from rdkit.Chem.Descriptors import ExactMolWt
        feature = np.array([ExactMolWt(Chem.MolFromSmiles(mol_string))], np.float32)
        return feature

    def description(self):
        return ['Mol weight']

class IntegrateFeature(Feature):
    """
    Integrate several features.
    """
    def __init__(self, features):
        super(IntegrateFeature, self).__init__()
        self.features = features

    def feature(self, mol_string):
        vector = None
        for feature in self.features:
            if vector is None:
                vector = feature.feature(mol_string)
            else:
                vector = np.concatenate([vector, feature.feature(mol_string)])
        return vector

    def description(self):
        feature_desc = []
        for feature in self.features:
            feature_desc = feature_desc + feature.description()
        return feature_desc


def process(feature, mol_strings, ys):
    xs = [feature.feature(mol_string) for mol_string in mol_strings]
    return np.array(xs), np.array(ys)
