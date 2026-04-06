import pickle

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)


def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data