from tqdm import tqdm

file = "/Users/allanjie/allan/data/ner-dataset-modified-dee/20k_mdee_gazz.txt"
out_file = "/Users/allanjie/allan/data/ner-dataset-modified-dee/20k_mdee_gazz.conll.txt"
sent = []
out = open(out_file, 'w', encoding='utf-8')
prev_label = 'O'
with open(file, 'r', encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        line = line.rstrip()
        # print(line)
        vals = line.split()
        word = '-'.join(vals[:-1])
        label = vals[-1]
        if word == ".":
            out.write(line + '\n\n')
            prev_label = 'O'
        else:
            if label != 'O':
                if prev_label == 'O':
                    label = "B-" + label
                else:
                    if prev_label[2:] == label:
                        label = "I-" + label
                    else:
                        label = "B-" + label
            prev_label = label
            out.write(word + ' ' + label + '\n')

out.close()