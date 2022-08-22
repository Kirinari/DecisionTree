import numpy as np
import pandas as pd
from collections import Counter


from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    def H(p): 
        return 1 - sum(p[p > 0] ** 2)

    def prob(target_vector): # расчёт вероятностей для критерия Джини
        L = len(target_vector)
        count_1 = np.sum(target_vector == 1)
        count_0 = L - count_1
        return np.array([count_0, count_1]) / L
    
    def gini(split_pair): # расчёт критерия Джини
        R_l = split_pair[0]
        R_r = split_pair[1]
        count_r = len(R_r)
        count_l = len(R_l)
        count_total = count_r + count_l
        return (count_l * H(prob(R_l)) + count_r * H(prob(R_r))) / count_total
    
    def split(arg): # разбиение на две подвыборки по всевозможным порогам
        pair, target = arg        
        cond = pair[0] <= pair[1]
        return [target[cond], target[~cond]]
    
    if (np.all(feature_vector == feature_vector.iloc[0])):
        return [], [], -1, -1
    
    ind = feature_vector.argsort() # сортируем по признаку с сохранением индексов
    feature_sorted = feature_vector.iloc[ind].unique()
    target_sorted = target_vector[ind]
    feature_shifted = np.append(feature_sorted[1:], [0]) # сдвигаем значения признака вправо
    thresholds = (feature_shifted + feature_sorted) / 2 # находим средние всех соседних значений
    thresholds = thresholds[:-1] # убираем лишний порог
    
     
    pairs = list(zip([feature_vector] * len(thresholds) , thresholds)) # создаём пары из списка признаков и списков (!) порогов
    targets = [target_vector] * len(thresholds) # создаём список из списков (!) целевой переменной
    splits = list(map(split, zip(pairs, targets))) # применяем ко всем объектам функцию split и делим ц. переменную
    
    ginis = np.array(list(map(gini, splits))) # по разделениям считаем Джини
    n_min = np.argmin(ginis) 
    gini_best = ginis[n_min] 
    threshold_best = thresholds[n_min] # находим минимальное значение Джини и соответствующий ему порог
    # и ни одного цикла ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    
    return thresholds, ginis, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {"depth" : 0}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        max_depth = self._max_depth
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X.iloc[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X.iloc[:, feature])
                clicks = Counter(sub_X.iloc[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = pd.Series(map(lambda x: categories_map[x], sub_X.iloc[:, feature]))
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = np.array(feature_vector < threshold)
                
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError 

        if (feature_best is None or node["depth"] == max_depth or
            len(sub_y) < self._min_samples_split or
               np.sum(split) < self._min_samples_leaf or
                   np.sum(np.logical_not(split)) < self._min_samples_leaf):
            node["type"] = "terminal"
            if feature_best is None:
                node["class"] = Counter(sub_y).most_common(1)
            else:
                node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        
        if (np.sum(split) != 0 and node["depth"] != max_depth):
            node['left_child'] = {'depth' : node['depth'] + 1}
            self._fit_node(sub_X.iloc[split, :], sub_y[split], node["left_child"])
            
        if (np.sum(np.logical_not(split)) != 0 and node["depth"] != max_depth):
            node['right_child'] = {'depth' : node['depth'] + 1}
            self._fit_node(sub_X.iloc[np.logical_not(split), :], sub_y[np.logical_not(split)], node["right_child"])
        
        return
        
        

    def _predict_node(self, x, node):
        while (node['type'] == 'nonterminal'):
            if 'categories_split' in node.keys():
                if x[node['feature_split']] in node['categories_split']:
                    if ('left_child' in node.keys()):
                        node = node['left_child']
                else:
                    if ('right_child' in node.keys()):
                        node = node['right_child']
            else:
                if (x[node['feature_split']] < node['threshold']):
                    if ('left_child' in node.keys()):
                        node = node['left_child']
                else:
                    if ('right_child' in node.keys()):
                        node = node['right_child']
        return node['class']

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in np.array(X):
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
