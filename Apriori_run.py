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


def data_preprocess(path):
    with open(path) as f:
        lines = f.readlines()
        data_lines = []
        for line in lines:
            line = line.strip('\n').split(',')
            # age
            if float(line[0])<=20:
                # line[0] = '{}under20'.format(line[0])
                age = 'junior'
            else:
                if float(line[0])<=60:
                    age = 'senior'
                else:
                    age = 'elder'
            if float(line[1])==1:
                gender = 'male'
            else:
                gender = 'female'
            chest_pain = 'chestpain{}'.format(line[2])
            if float(line[3])<=120:
                blood_pressure = 'bloodpressure<=120'
            else:
                if float(line[3])<140:
                    blood_pressure = '120<bloodPressure<=140'
                else:
                    blood_pressure = 'bloodPressure>140'

            if float(line[4])<200:
                chol = 'chol<200'
            else:
                if float(line[4])<220:
                    chol = '200<chol<=220'
                else:
                    chol = '220<chol'
            if float(line[5]):
                fbs = 'fbs>120'
            else:
                fbs = 'fbs<=120'

            restecg = 'restecg{}'.format(line[6])
            thalach = 'higher' if (220-float(line[0]))>float(line[7]) else 'lower'
            exang = 'exang{}'.format(line[8])
            slope = 'slope{}'.format(line[10])
            ca = 'ca{}'.format(line[11])
            thal = 'thal{}'.format(line[12])
            num = 'num=1' if float(line[-1]) else 'num=0'

            data_lines.append([age, gender, chest_pain, blood_pressure, chol, fbs, restecg,
                               thalach, exang, slope, ca, thal, num])
    f.close()
    return data_lines


if __name__ == '__main__':
    # dataset = [
    #     [1, 2, 5], [2, 3], [1, 2, 4], [1, 3],
    #     [2, 4], [2, 3], [1, 3], [1, 2, 3, 5], [1, 2, 3]
    # ]
    min_support = 50
    min_confidence = 70
    # dataset = data_load('./groceries.csv')
    dataset = data_preprocess('processed.cleveland.data')

    apriori_master = Apriori(dataset, min_support, min_confidence)
    apriori_master.apriori_solve()
    apriori_master.association_rule_generate()

pass