from Apriori import Apriori


def data_load(path):
    data = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            data_line = []
            for element in line:
                if element == '' or element == '\n':
                    continue
                data_line.append(int(element))
            data.append(data_line)
    f.close()
    return data


if __name__ == '__main__':
    # dataset = [
    #     [1, 2, 5], [2, 3], [1, 2, 4], [1, 3],
    #     [2, 4], [2, 3], [1, 3], [1, 2, 3, 5], [1, 2, 3]
    # ]
    min_support = 70
    min_confidence = 70
    dataset = data_load('./forests.txt')

    apriori_master = Apriori(dataset, min_support, min_confidence)
    apriori_master.apriori_solve()
    apriori_master.association_rule_generate()

pass