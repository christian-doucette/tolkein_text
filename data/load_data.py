import urllib.request
import re
from bs4 import BeautifulSoup


# Sets the apostraphes within contractions to ', and others to "
def parse_and_fix_quotes(input):
    text = input.getText()
    text = re.sub('[\'\"`“”‘’]', '\"', text)
    text = re.sub('(?<=\w)[\"](?=\w)', '\'', text)

    text = re.sub('[\n\t]', ' ', text)

    return text



# loads and parses html text for LOTR book num
def load_from_url(num):
    fp = urllib.request.urlopen(f"http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_{num}__en.htm")
    mybytes = fp.read()
    mystr = mybytes.decode("cp1252")

    soup = BeautifulSoup(mystr, 'html.parser')
    p_tags = soup.find_all('p')


    start = [109, 28, 27][num-1]
    end = [3955, 2942, 2449][num-1]
    p_tags_processed_text = list(map(parse_and_fix_quotes, p_tags))[start:end]
    #p_tags_unprocessed = list(map(lambda x: x.getText(), p_tags))[111]

    full_text = " ".join(p_tags_processed_text)
    #Fellowship Start: 109, End: 3955
    #Two towers Start: 28,  End: 2942
    #ROTK       Start: 27,  End: 2449
    return full_text



full_text  = load_from_url(3)
#print(full_text)

"""
with open("rotk.txt", "w") as text_file:
    text_file.write(full_text)
print('Done')
"""
