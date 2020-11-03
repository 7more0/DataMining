import more_itertools
# data struct
# input: list of affairs
# Itemsets: element list of itemsets and frequency count of (k-1)-itemset
#          contain [[itemset], [frequency]]


class Apriori:
    def __init__(self, data, min_support, min_confidence):
        self.Data = data
        self.min_sup = round(min_support*len(data)/100)
        self.Itemsets = []
        self.min_support = min_support
        self.min_confidence = min_confidence
        # build itemset L1
        self.Items = []
        for aff in data:
            for item in aff:
                if [item] not in self.Items:
                    self.Items.append([item])
        frequency = self.frequency_calc(self.Items)
        self.Itemsets = [self.Items, frequency]
        self.AllFrequentItemsets = []           # iterating itemsets
        self.AllFrequentItemsets.append(self.Itemsets)      # itersets history

    def apriori_gen(self, pre_itemset):
        # generate k-itemset with (k-1)-itemset
        itemset_k = []
        k = len(pre_itemset[0][0])+1
        for itemset_a in pre_itemset[0]:
            for itemset_b in pre_itemset[0]:
                new_set = set(itemset_a+itemset_b)
                if new_set not in itemset_k and len(new_set) == k:
                    itemset_k.append(new_set)
        for set_index in range(len(itemset_k)):
            itemset_k[set_index] = list(itemset_k[set_index])
        return itemset_k

    def frequency_calc(self, itemsets):
        # calculate all frequency of itemsets in itemsets
        # return result as a list with same order
        freqency = []
        data = self.Data
        for itemset in itemsets:
            freq = 0
            for affair in data:
                fail = False
                for item in itemset:
                    if item not in affair:
                        fail = True
                        break
                if not fail:
                    freq += 1
            freqency.append(freq)
            # debug
            print('{}: {}'.format(itemset, freq))
        return freqency

    def apriori_solve(self):
        # do apriori algorithm
        i = 1
        while True:
            itemset_i = self.apriori_gen(self.Itemsets)
            itemset_frequency = self.frequency_calc(itemset_i)
            Itemset_i = []
            Itemset_frequency = []
            # travel and record all frequent itemsets
            for index in range(len(itemset_i)):
                if itemset_frequency[index] >= self.min_sup:
                    Itemset_i.append(itemset_i[index])
                    Itemset_frequency.append((itemset_frequency[index]))
            i += 1
            if len(Itemset_i) == 0:
                # process finished at k=i+1 stage
                print('Find no frequent {}-itemset'.format(i))
                break
            else:
                self.Itemsets = [Itemset_i, Itemset_frequency]
                self.AllFrequentItemsets.append(self.Itemsets)

        # # print result
        # print('================================')
        # print('All frequent itemsets:')
        # print('================================')
        # for itemset_index in range(len(self.Itemsets[0])):
        #     print('{}: {}'.format(self.Itemsets[0][itemset_index], self.Itemsets[1][itemset_index]))
        # print('================================')
        return self.Itemsets[-1]

    def subset_generate(self, p_set):
        # generate all true-subsets of p_set
        subsets = list(more_itertools.powerset(p_set))
        subsets_return = []
        for subset in subsets:
            if len(subset) == 0 or len(subset) == len(p_set):
                continue
            else:
                subsets_return.append(list(subset))
        return subsets_return

    def association_rule_generate(self):
        # generate association rules based on frequent itemsets generated by apriori_solve()
        association_rules = []
        for sets_ind in range(1, len(self.AllFrequentItemsets)):
            sets = self.AllFrequentItemsets[sets_ind]
            # travel all frequent itemsets
            for set_index in range(len(sets[0])):
                set_frequency = sets[1][set_index]
                subsets = self.subset_generate(sets[0][set_index])
                subset_frequency = self.frequency_calc(subsets)
                # store all rules as [[[set A],[set B]], support, confidence]
                rule_support = set_frequency*100/len(self.Data)
                # travel all subsets
                for subset_index in range(len(subsets)):
                    subset = subsets[subset_index]
                    res_set = list(set(sets[0][set_index])-set(subset))
                    rule_confidence = set_frequency*100/subset_frequency[subset_index]
                    # debug
                    print('{} => {}; support: {:.2f}%  confidence: {:.2f}%'.
                          format(subset, res_set, rule_support, rule_confidence))
                    association_rules.append([[subset, res_set],
                                              rule_support, rule_confidence])

        # print result
        print('===================================================')
        print('All frequent itemsets generated:')
        print('===================================================')
        for itermsets_ind in range(1, len(self.AllFrequentItemsets)):
            itemsets = self.AllFrequentItemsets[itermsets_ind]
            for sets_ind in range(len(itemsets)):
                print('{}: {}'.format(itemsets[0][sets_ind], itemsets[1][sets_ind]))
        print('===================================================')
        print('All association rules generated:')
        print('===================================================')
        rule_count = 0
        for rule in association_rules:
            if rule[2] >= self.min_confidence and rule[1] >= self.min_support:
                print('{} => {}; support: {:.0f}%  confidence: {:.0f}%'.
                      format(rule[0][0], rule[0][1], rule[1], rule[2]))
                rule_count += 1
        print('=============== {} rules in total ================='.format(rule_count))




