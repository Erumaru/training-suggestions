import csv

class Studio:
    def __init__(self, id, name):
        self.id = id
        self.name = name

studios = []

def parse():
    with open('data/studios.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for row in reader:
            id = row[0]
            name = ' '.join(row[1:len(row)])
            studios.append(Studio(id, name))
