import urllib.request
import re
from bs4 import BeautifulSoup


def parse_and_fix_quotes(input):
    text = input.getText()
    text = re.sub('[\'\"‘`]', '\'', text)
    text = re.sub('[\n\t]', ' ', text)

    return text



# loads and parses html text for Fellowship of the Ring
def load_from_url(url):
    fp = urllib.request.urlopen("http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_3__en.htm")
    mybytes = fp.read()
    mystr = mybytes.decode("latin-1")

    soup = BeautifulSoup(mystr, 'html.parser')
    p_tags = soup.find_all('p')

    p_tags_processed_text = list(map(parse_and_fix_quotes, p_tags))[27:2449]
    #p_tags_unprocessed = list(map(lambda x: x.getText(), p_tags))[111]

    full_text = " ".join(p_tags_processed_text)
    #Fellowship Start: 109, End: 3955
    #Two towers Start: 24,  End: 2942
    #ROTK       Start: 27,  End: 2449
    return full_text



full_text  = load_from_url('test')
print(full_text)


"""
with open("rotk.txt", "w") as text_file:
    text_file.write(full_text)
print('Done')
"""
