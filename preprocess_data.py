import urllib.request
from bs4 import BeautifulSoup
import re



def parse_and_clean(page):
    text = page.getText()
    text = re.sub('Mr.', 'Mr ', text) #removes all single/double quotes
    text = re.sub('[\'\"‘]', '', text) #removes all single/double quotes
    text = re.sub(r"(\.|\,|\(|\)|\;|\:)", lambda x: f' {x.group(1)} ', text)
    text = re.sub('[^0-9a-zA-Z.,:;()]+', ' ', text) #replaces all remaining non-alphanumeric/punctuationcharacters with space
    text = text.lower()
    return text


def load_from_url(url):
    fp = urllib.request.urlopen("http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_1__en.htm")
    mybytes = fp.read()
    mystr = mybytes.decode("latin-1")

    soup = BeautifulSoup(mystr, 'html.parser')
    p_tags = soup.find_all('p')
    p_tags_processed_text = list(map(parse_and_clean, p_tags))[109:3955]
    full_text = "".join(p_tags_processed_text)
    #Fellowship Start: 109, End: 3954 (inclusive)
    return full_text.split()
#print(pages_text)
