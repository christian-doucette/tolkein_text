# Tolkein Text
Tolkein Text is a personal project that I created to practice implementing Machine Learning, specifically NLP, in Pytorch. It uses a Neural Network Language Model to generate text based off *The Fellowship of the Ring*.

### Some Examples
"But the road ran up again to the top."  
"It is probably that this evening has gone now into light."  
"Hope I wish to you."

These sentences definitely capture the feel of Tolkein's writing, but are totally original!

### Preprocessing
The preprocessing stage can be broken down into these steps:
1. Load html file from http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_1__en.htm
2. Parse with BeautifulSoup to get just the text
3. Remove capitalization and some punctuation with Regex
4. Build vocabulary that associates each word to a unique id. This includes word_to_id (dictionary), and id_to_word (direct access table)
5. Map each word in the text to its id
6. Add each word_id in the text to dataset as a label with the 19 word_ids before it as its feature
7. Create DataLoader with batch size of 32. This will allow me to cycle through all the data by random sets of 32 feature/label pairs.

### Network Architecture
The network has the following architecture:
Input -> Embedding layer -> LSTM layers -> Dropout layer -> Fully Connected Linear Layer -> Output

Input is the 19 preceding word_ids I mentioned before.  
Output is a list of vocab_size values, where a higher value means that that word is more likely to occur next.

### Network Training
Using the network architecture stated above and cross entropy loss, I find the weights that minimize loss on the training data. These weights are calculated with gradient descent using Pytorch's Autograd package.

### Text Generation
By applying the softmax function to the output of the network, I get a probability distribution over all remaining words. Then, I take a weighted random choice of these to decide the next word.
