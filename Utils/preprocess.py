import sys

from Utils import metric

label_dict = {'CP_START': 2, 'CP': 1,
              'CC': 3, 'SEP': 4, 'OTHERS': 5, 'NONE': 0}


# function that returns the split sentences from
# openie labels
def get_sentences_from_openie_labels(input_file):
    file = open(input_file)
    sentence = ""
    predictions = []
    all_predictions = []
    for line_in_file in file:
        line_in_file = line_in_file[:-1]
        if line_in_file.startswith("NONE") or line_in_file.startswith("CP_START") or line_in_file.startswith(
                "CP") or line_in_file.startswith("CC") or line_in_file.startswith("OTHERS") or line_in_file.startswith(
            "SEP"):
            split_line = line_in_file.split()
            for label in split_line:
                predictions.append(label_dict[label])
            all_predictions.append(predictions)
            predictions = []
        elif line_in_file == "":
            continue
        else:
            if len(all_predictions) != 0:
                coords = metric.get_coords(all_predictions)
                print("sentence is: " + sentence + "\n, coords are: " + str(coords))
                words = sentence.split()
                split_sentences, conj_words, sentences_indices = coords_to_sentences(
                    coords, words)
                roots, parent_mapping, child_mapping = coords_to_tree(coords, words)

                discource_tree = construct_discource_tree(sentences_indices, roots, parent_mapping, coords)
                print("discource tree is: "+str(discource_tree))

                print("split_sentences are: " + str(split_sentences) +
                      ",\n conj_words are: " + str(conj_words) + ",\n sentences_indices are: " + str(sentences_indices))

                all_predictions = []
            sentence = line_in_file


def construct_discource_tree(sentences_indices, roots, parent_mapping, coords):
    discource_tree = {}
    count = 0
    if len(parent_mapping) != 0:
        for child in parent_mapping:
            coords_for_child = coords[child]
            conjuncts_for_child = coords_for_child.conjuncts

            for conjunct_for_child in conjuncts_for_child:
                for i in range(0, len(sentences_indices)):
                    if (conjunct_for_child[0] in sentences_indices[i] and
                            conjunct_for_child[1] in sentences_indices[i]):
                        discource_tree[i] = count
                        break
            count = count + 1
        for r in roots:
            coords_for_root = coords[r]
            conjuncts_for_root = coords_for_root.conjuncts
            
            for conjunct_for_root in conjuncts_for_root:
                for i in range(0, len(sentences_indices)):
                    if conjunct_for_root[0] in sentences_indices[i] and conjunct_for_root[1] in sentences_indices[i]:
                        if i not in discource_tree:
                            discource_tree[i] = count
                            break
    return discource_tree


def coords_to_tree(conj_coords, words):
    print("length of words: " + str(len(words)))
    word_sentences = []
    # print("length of conj_coords: "+str(len(conj_coords)))
    for k in list(conj_coords):
        if conj_coords[k] is None:
            conj_coords.pop(k)
    try:
        for k in list(conj_coords):
            print(conj_coords[k].cc)
            if words[conj_coords[k].cc] in ['nor', '&']:  # , 'or']:
                conj_coords.pop(k)
    except:
        print("problem with conj_coords")
        return [], [], []

    # print("words are: "+str(words))
    # print("length of words: "+str(len(words)))

    num_coords = len(conj_coords)
    remove_unbreakable_conjuncts(conj_coords, words)

    roots, parent_mapping, child_mapping = get_tree(conj_coords)
    print("after get_tree, roots: " + str(roots) + ", parent_mapping: " +
          str(parent_mapping) + ", child_mapping: " + str(child_mapping))
    return roots, parent_mapping, child_mapping


def coords_to_sentences(conj_coords, words):
    sentence_indices = []
    for i in range(0, len(words)):
        sentence_indices.append(i)

    conj_words = []
    for k in list(conj_coords):
        for conjunct in conj_coords[k].conjuncts:
            conj_words.append(' '.join(words[conjunct[0]:conjunct[1] + 1]))

    roots, parent_mapping, child_mapping = coords_to_tree(conj_coords, words)
    print("roots: " + str(roots) + ", parent_mapping: " +
          str(parent_mapping) + ", child_mapping: " + str(child_mapping))

    q = list(roots)

    sentences = []
    count = len(q)
    new_count = 0

    conj_same_level = []

    while (len(q) > 0):

        conj = q.pop(0)
        count -= 1
        conj_same_level.append(conj)

        for child in child_mapping[conj]:
            q.append(child)
            new_count += 1

        if count == 0:
            get_sentences(sentences, conj_same_level,
                          conj_coords, sentence_indices)
            # print("sentences list is: "+str(sentences))
            count = new_count
            new_count = 0
            conj_same_level = []

    try:
        word_sentences = [' '.join([words[i] for i in sorted(sentence)]) for sentence in sentences]

    except:
        print("exception occurred for " + str(words))
    return word_sentences, conj_words, sentences
    # return '\n'.join(word_sentences) + '\n'


def get_tree(conj):
    parent_child_list = []

    child_mapping, parent_mapping = {}, {}

    for key in conj:
        assert conj[key].cc == key
        parent_child_list.append([])
        for k in conj:
            if conj[k] is not None:
                if is_parent(conj[key], conj[k]):
                    parent_child_list[-1].append(k)

        child_mapping[key] = parent_child_list[-1]

    parent_child_list.sort(key=list.__len__)

    for i in range(0, len(parent_child_list)):
        for child in parent_child_list[i]:
            for j in range(i + 1, len(parent_child_list)):
                if child in parent_child_list[j]:
                    parent_child_list[j].remove(child)

    for key in conj:
        for child in child_mapping[key]:
            parent_mapping[child] = key

    roots = []
    for key in conj:
        if key not in parent_mapping:
            roots.append(key)

    return roots, parent_mapping, child_mapping


def is_parent(parent, child):
    min = child.conjuncts[0][0]
    max = child.conjuncts[-1][-1]

    for conjunct in parent.conjuncts:
        if conjunct[0] <= min and conjunct[1] >= max:
            return True
    return False


def get_sentences(sentences, conj_same_level, conj_coords, sentence_indices):
    for conj in conj_same_level:

        if len(sentences) == 0:

            for conj_structure in conj_coords[conj].conjuncts:
                sentence = []
                for i in range(conj_structure[0], conj_structure[1] + 1):
                    sentence.append(i)
                sentences.append(sentence)

            min = conj_coords[conj].conjuncts[0][0]
            max = conj_coords[conj].conjuncts[-1][-1]

            for sentence in sentences:
                for i in sentence_indices:
                    if i < min or i > max:
                        sentence.append(i)

        else:
            to_add = []
            to_remove = []

            for sentence in sentences:
                if conj_coords[conj].conjuncts[0][0] in sentence:
                    sentence.sort()

                    min = conj_coords[conj].conjuncts[0][0]
                    max = conj_coords[conj].conjuncts[-1][-1]

                    for conj_structure in conj_coords[conj].conjuncts:
                        new_sentence = []
                        for i in sentence:
                            if i in range(conj_structure[0], conj_structure[1] + 1) or i < min or i > max:
                                new_sentence.append(i)

                        to_add.append(new_sentence)

                    to_remove.append(sentence)

            for sent in to_remove:
                sentences.remove(sent)
            sentences.extend(to_add)


def remove_unbreakable_conjuncts(conj, words):
    unbreakable_indices = []

    unbreakable_words = ["between", "among", "sum", "total", "addition", "amount", "value", "aggregate", "gross",
                         "mean", "median", "average", "center", "equidistant", "middle"]

    for i, word in enumerate(words):
        if word.lower() in unbreakable_words:
            unbreakable_indices.append(i)

    to_remove = []
    span_start = 0

    for key in conj:
        span_end = conj[key].conjuncts[0][0] - 1
        for i in unbreakable_indices:
            if span_start <= i <= span_end:
                to_remove.append(key)
        span_start = conj[key].conjuncts[-1][-1] + 1

    for k in set(to_remove):
        conj.pop(k)


def get_sentences_from_tree_labels():
    return 0


def convert_seq_labels_to_tree():
    # get the parent to child mapping

    return 0


def preprocess_input(arg1, arg2):
    f = open(arg1)

    o = open(arg2, "w")

    for line in f:
        line = line[:-1]
        if line.startswith("NONE") or line.startswith("CP_START") or line.startswith("CP") or line.startswith(
                "CC") or line.startswith("OTHERS") or line == "":
            continue
        else:
            o.write("#" + line + "\n")
            o.write("JARGON\n")
    o.close()


if __name__ == '__main__':
    get_sentences_from_openie_labels(
        "/media/prajna/Files11/bits/faculty/project/labourlaw_kg/LegalIE/data/ptb-test_split_small.labels")
