import sys

# from ..Utils import metric
import metric

label_dict = {'CP_START': 2, 'CP': 1,
              'CC': 3, 'SEP': 4, 'OTHERS': 5, 'NONE': 0}


# function that returns the split sentences from
# openie labels
def get_sentences_from_openie_labels(input_file, output_file):
    file = open(input_file)
    o = open(output_file, "w")
    o1 = open(output_file.replace(".txt", "") + ".conj", "w")
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
                o.write("#" + sentence + "\n")
                o1.write(sentence + "\n")
                print("sentence is: " + sentence + "\n, coords are: " + str(coords))
                words = sentence.split()
                split_sentences, conj_words, sentences_indices = coords_to_sentences(
                    coords, words)
                roots, parent_mapping, child_mapping = coords_to_tree(coords, words)
                print("parent_mapping: " + str(parent_mapping))
                for ss in split_sentences:
                    o1.write(ss+"\n")
                o1.write("\n")

                discource_tree = construct_discource_tree(sentences_indices, roots, parent_mapping, coords)
                discource_tree_inverse = {}
                print("discource tree is: " + str(discource_tree))
                for sent_index in discource_tree:
                    if discource_tree[sent_index] not in discource_tree_inverse:
                        discource_tree_inverse[discource_tree[sent_index]] = []
                        discource_tree_inverse[discource_tree[sent_index]].append(sent_index)
                    else:
                        discource_tree_inverse[discource_tree[sent_index]].append(sent_index)
                print(str(discource_tree_inverse))

                count = 0
                partial_coordination_str = ""
                while count in discource_tree_inverse:
                    sent_indices = discource_tree_inverse[count]
                    partial_coordination_str = get_coordination_string(sent_indices, partial_coordination_str,
                                                                       split_sentences)
                    count = count + 1
                print(partial_coordination_str)
                if partial_coordination_str == "":
                    partial_coordination_str = "NONE"

                o.write(partial_coordination_str + "\n\n")

                print("split_sentences are: " + str(split_sentences) +
                      ",\n conj_words are: " + str(conj_words) + ",\n sentences_indices are: " + str(sentences_indices))

                all_predictions = []
            sentence = line_in_file
    if len(all_predictions) != 0:
        coords = metric.get_coords(all_predictions)
        o.write("#" + sentence + "\n")
        o1.write(sentence + "\n")
        print("sentence is: " + sentence + "\n, coords are: " + str(coords))
        words = sentence.split()
        split_sentences, conj_words, sentences_indices = coords_to_sentences(
            coords, words)
        roots, parent_mapping, child_mapping = coords_to_tree(coords, words)
        print("parent_mapping: " + str(parent_mapping))

        for ss in split_sentences:
            o1.write(ss + "\n")
        o1.write("\n")

        discource_tree = construct_discource_tree(sentences_indices, roots, parent_mapping, coords)
        discource_tree_inverse = {}
        print("discource tree is: " + str(discource_tree))
        for sent_index in discource_tree:
            if discource_tree[sent_index] not in discource_tree_inverse:
                discource_tree_inverse[discource_tree[sent_index]] = []
                discource_tree_inverse[discource_tree[sent_index]].append(sent_index)
            else:
                discource_tree_inverse[discource_tree[sent_index]].append(sent_index)
        print(str(discource_tree_inverse))

        count = 0
        partial_coordination_str = ""
        while count in discource_tree_inverse:
            sent_indices = discource_tree_inverse[count]
            partial_coordination_str = get_coordination_string(sent_indices, partial_coordination_str,
                                                               split_sentences)
            count = count + 1
        print(partial_coordination_str)
        if partial_coordination_str == "":
            partial_coordination_str = "NONE"

        o.write(partial_coordination_str + "\n\n")

        print("split_sentences are: " + str(split_sentences) +
              ",\n conj_words are: " + str(conj_words) + ",\n sentences_indices are: " + str(sentences_indices))

        all_predictions = []
    o.close()
    o1.close()


def get_coordination_string(sent_indices, partial_coordination_str, split_sentences):
    coordination_str = "COORDINATION("
    # print(sent_indices)
    # print(split_sentences)
    count = 0
    for indexes in sent_indices:
        if count == len(sent_indices) - 1:
            coordination_str = coordination_str + "\" " + split_sentences[indexes] + "\""
        else:
            coordination_str = coordination_str + "\" " + split_sentences[indexes] + "\" , "
        count = count + 1
    coordination_str = coordination_str + " " + partial_coordination_str + ")"
    return coordination_str


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
    else:
        for i in range(0, len(sentences_indices)):
            discource_tree[i] = 0
    return discource_tree


def coords_to_tree(conj_coords, words):
    # print("length of words: " + str(len(words)))
    word_sentences = []
    # print("length of conj_coords: "+str(len(conj_coords)))
    for k in list(conj_coords):
        if conj_coords[k] is None:
            conj_coords.pop(k)
    try:
        for k in list(conj_coords):
            # print(conj_coords[k].cc)
            if words[conj_coords[k].cc] in ['nor', '&']:  # , 'or']:
                conj_coords.pop(k)
    except:
        # print("problem with conj_coords")
        return [], [], []

    # print("words are: "+str(words))
    # print("length of words: "+str(len(words)))

    num_coords = len(conj_coords)
    remove_unbreakable_conjuncts(conj_coords, words)

    roots, parent_mapping, child_mapping = get_tree(conj_coords)
    # print("after get_tree, roots: " + str(roots) + ", parent_mapping: " +
    #      str(parent_mapping) + ", child_mapping: " + str(child_mapping))
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
    # print("roots: " + str(roots) + ", parent_mapping: " +
    #      str(parent_mapping) + ", child_mapping: " + str(child_mapping))

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


def get_sentences_from_tree_labels(model, tree_label):
    if tree_label == "NONE":
        return [""]
    count = tree_label.count("COORDINATION")
    # Removing " )) from the end
    if count >= 1:
        tree_label = tree_label[:-(count + 2)]
    # if(model == "OpenIE"):
    #     sentences = tree_label.split("\" , \"")
    # else:
    #     sentences = tree_label.split("\", \"")
    sentences = tree_label.split("\" , \"")
    new_sentenes = []
    if model == "BART":
        for s in sentences:
            if "\" COORDINATIONAL" in s:
                s1 = s.split("\" COORDINATIONAL(\"")
                for ss in s1:
                    ss = ss.replace("COORDINATION(\" ", "")
                    new_sentenes.append(ss)
            elif "COORDINATIONAL" in s:
                s1 = s.split("COORDINATIONAL(\"")
                for ss in s1:
                    ss = ss.replace("COORDINATION(\" ", "")
                    new_sentenes.append(ss)
            elif "\" COORDINATION" in s:
                s1 = s.split("\" COORDINATION(\"")
                for ss in s1:
                    ss = ss.replace("COORDINATION(\" ", "")
                    new_sentenes.append(ss)
            elif "COORDINATION" in s:
                s1 = s.split("COORDINATION(\"")
                for ss in s1:
                    ss = ss.replace("COORDINATION(\" ", "")
                    new_sentenes.append(ss)
            else:
                s = s.replace("COORDINATION(\" ", "")
                new_sentenes.append(s)
    elif model == "T5" or model == "OpenIE":
        for s in sentences:
            if "\" COORDINATION" in s:
                s1 = s.split("\" COORDINATION(\"")
                for ss in s1:
                    ss = ss.replace("COORDINATION(\" ", "")
                    new_sentenes.append(ss)
            elif "COORDINATION" in s:
                s1 = s.split("COORDINATION(\"")
                for ss in s1:
                    ss = ss.replace("COORDINATION(\"", "")
                    new_sentenes.append(ss)
            else:
                s = s.replace("COORDINATION(\" ", "")
                new_sentenes.append(s)
    else:
        print("Invalid model name")
        sys.exit(0)
    return new_sentenes


def convert_discource_tree_to_conj(model, input_file, output_file):
    f = open(input_file)
    o = open(output_file, "w")
    sentence = ""
    prediction = ""
    for line in f:
        if line.startswith("Input: "):
            sentence = line.replace("Input: ", "")[:-1]
        elif line.startswith("Prediction: "):
            prediction = line.replace("Prediction: ", "")[:-1]
            new_sentences = get_sentences_from_tree_labels(model, prediction)
            o.write(sentence + "\n")
            for sentences in new_sentences:
                if sentences.strip()=="":
                    continue
                o.write(sentences.strip() + "\n")
            o.write("\n")
    o.close()


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
    if len(sys.argv) != 4:
        print("Usage: python3 preprocess.py T5 <input_file> <output_file>")
        exit(0)
    convert_discource_tree_to_conj(sys.argv[1], sys.argv[2], sys.argv[3])
    # convert_discource_tree_to_conj(
    #     "/media/prajna/Files11/bits/faculty/project/labourlaw_kg/LegalIE/data/CoordinationDataSet/Prediction_BART_Coordination.txt",
    #     "/media/prajna/Files11/bits/faculty/project/labourlaw_kg/LegalIE/data/CoordinationDataSet/Prediction_BART_Coordination.conj")
    # get_sentences_from_openie_labels(
    #   "/media/prajna/Files11/bits/faculty/project/labourlaw_kg/LegalIE/data/ptb-test_split.labels",
    #  "/media/prajna/Files11/bits/faculty/project/labourlaw_kg/LegalIE/data/coordination_tree_encoding")
