subordination_labels = {'SUB/ELABORATION': 1,
                        'SUB/SPATIAL': 2,
                        'SUB/ATTRIBUTION': 3,
                        'SUB/UNKNOWN_SUBORDINATION': 4,
                        'CO/CONTRAST': 5,
                        'CO/LIST': 6,
                        'CO/DISJUNCTION': 7,
                        'SUB/CAUSE': 8,
                        'SUB/CONDITION': 9,
                        'SUB/PURPOSE': 10}

def get_sentences_from_tree_labels_subordination(tree_label, leaf_sentences):
    print("tree_label is: "+tree_label+"\n")
    org = tree_label
    tree_label1 = tree_label
    for relation in subordination_labels:
        print(relation)
        # remove root node
        if tree_label.startswith(relation):
            tree_label1 = tree_label.replace(relation, "", 1)
            print("tree_label1 after removing relation: "+tree_label1)
            break
    if org == tree_label1:
        print("final: "+tree_label.strip('(\")'))
        leaf_sentences.append(tree_label.strip('(\")'))
    else:
        sentences = tree_label1.split("\",")
        for s in sentences:
            get_sentences_from_tree_labels_subordination(s.strip(), leaf_sentences)

if __name__ == '__main__':

    #label = ("SUB/ELABORATION(\"Women account for the vast majority of those.\", SUB/ELABORATION(\"Those have lost "
    #                  "their jobs (24.7 million.\",\"Those have lost their jobs.\"))")

    label = "SUB/CONDITION(\"Any contribution of minimum amount in any year is not invested.\", SUB/ELABORATION(\"This was then.\",\"The account will be deactivated.\"))"
    leaf_sentences = []

    get_sentences_from_tree_labels_subordination(label, leaf_sentences)
    print(leaf_sentences)
