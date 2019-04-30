# 
# @author: Allan
#

def process(filename:str, out:str):
    fres = open(out, 'w', encoding='utf-8')
    print(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        heads = []
        deps =[]
        labels = []
        prev_label = "O"
        prev_raw_label = ""
        for line in f.readlines():
            line = line.rstrip()
            # print(line)
            if line.startswith("#"):
                prev_label = "O"
                prev_raw_label = ""
                continue
            if line == "":
                idx = 1
                for w, h, dep, label in zip(words, heads, deps, labels):
                    if dep == "sentence":
                        dep = "root"
                    fres.write("{}\t{}\t_\t_\t_\t_\t{}\t{}\t_\t_\t{}\n".format(idx, w, h, dep, label))
                    idx += 1
                fres.write('\n')
                words = []
                heads = []
                deps = []
                labels = []
                prev_label = "O"
                continue
            #1	West	_	NNP	NNP	_	5	compound	_	_	B-MISC
            vals = line.split()
            idx = vals[0]
            word = vals[1]
            head = vals[8]
            dep_label = vals[10]
            label = vals[12]

            if label.startswith("("):
                if label.endswith(")"):
                    label = "B-" + label[1:-1]
                else:
                    label = "B-" + label[1:]
            elif label.startswith(")"):
                label = "I-" + label[:-1]
            else:
                if prev_label == "O":
                    label = "O"
                else:
                    if prev_raw_label.endswith(")"):
                        label = "O"
                    else:
                        label = "I-" + prev_label[2:]

            words.append(word)
            heads.append(head)
            labels.append(label)
            deps.append(dep_label)
            prev_label = label
            prev_raw_label = vals[12]
    fres.close()




# process("data/semeval10t1/en.train.txt", "data/semeval10t1/train.sd.conllx")
# process("data/semeval10t1/en.devel.txt", "data/semeval10t1/dev.sd.conllx")
# process("data/semeval10t1/en.test.txt", "data/semeval10t1/test.sd.conllx")

lang = "it"
folder="sem" + lang
process("data/"+folder+"/"+lang+".train.txt", "data/"+folder+"/train.sd.conllx")
process("data/"+folder+"/"+lang+".devel.txt", "data/"+folder+"/dev.sd.conllx")
process("data/"+folder+"/"+lang+".test.txt", "data/"+folder+"/test.sd.conllx")