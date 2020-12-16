import re
import json



#===========================================#
#        Load/Processing Functions          #
#===========================================#


# cleans up text by removing extraneous characters
def regex_parse(input):
    text = re.sub('Mr\.', 'Mr ', input)                                          # Removes changes Mr. to Mr to avoid period confusion
    text = re.sub(r"(\.|\,|\;|\:|\!|\?)", lambda x: f' {x.group(1)} ', text)    # Adds space on both sides of punctuation
    text = re.sub(re.compile('[^\w,.!?:;\']+', re.UNICODE), ' ', text)          # Replaces all remaining non-alphanumeric/punctuation with space

    return text



# loads full Lord of the Rings text
def load_lotr():
    with open('../data/fotr.txt', 'r') as file:
        fotr = regex_parse(file.read())

    with open('../data/tt.txt', 'r') as file:
        tt = regex_parse(file.read())

    with open('../data/rotk.txt', 'r') as file:
        rotk = regex_parse(file.read())

    lotr_full = fotr + tt + rotk
    return lotr_full.split()








#===========================================#
#     Stores Always-Capitalized Words       #
#===========================================#


with open('../trained_model/word_to_id.json') as json_file:
    word_to_id = json.load(json_file)

should_capitalize = {word: True for word in word_to_id}


lotr_full_text = load_lotr()

for word in lotr_full_text:
    if word in should_capitalize:
        should_capitalize[word] = False





#===========================================#
#               Manual Fixes                #
#===========================================#

should_capitalize['merry']     = True
should_capitalize['balin']     = True
should_capitalize['moria']     = True
should_capitalize['i']         = True
should_capitalize['bill']      = True
should_capitalize['orthanc']   = True
should_capitalize['butterbur'] = True




#===========================================#
#     Saves in always_capitalized.json      #
#===========================================#

with open('../trained_model/always_capitalized.json', 'w') as fp:
    json.dump(should_capitalize, fp, indent=4)
