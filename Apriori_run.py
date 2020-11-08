from Apriori import Apriori


def data_load(path):
    data = []
    data_dict = {}
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(',')
            data_line = []
            for element in line:
                if element == '' or element == '\n':
                    continue
                # data_line.append(int(element))
                data_line.append(element)
            data.append(data_line)
    f.close()
    return data


if __name__ == '__main__':
    min_support = 30
    min_confidence = 70
    dataset = data_load('./groceries.csv')

    apriori_master = Apriori(dataset, min_support, min_confidence)
    apriori_master.apriori_solve()
    apriori_master.association_rule_generate()

pass