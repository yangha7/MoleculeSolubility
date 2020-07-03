import csv
import os

class DatasetReader:
    def __init__(self):
        return

    def read(self, csv_path):
        """

        :param csv_path: path to the CSV dataset.
        :return: Molecules (SMILES) and the desired property.
        """
        return

class DelaneyReader(DatasetReader):
    def __init__(self):
        super(DelaneyReader, self).__init__()
        return

    def read(self, csv_path):
        # header = ['smiles', 'logSolubility']
        if not os.path.exists(csv_path):
            return [], []

        smiles = []
        property = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            reader.__next__() # Skip header
            for row in reader:
                smiles.append(row['smiles'])
                property.append(float(row['logSolubility']))

        return smiles, property

if __name__ == '__main__':
    reader = DelaneyReader()
    x, y = reader.read("./dataset/chem/delaney.csv")
    print(x, y)

