# Tolkein Text
Tolkein Text is a personal project that I created to practice implementing Machine Learning, specifically NLP, in Pytorch. It uses a Neural Network Language Model to generate text based off *The Fellowship of the Ring*.

### Some Examples
"But the road ran up again to the top."  
"It is probably that this evening has gone now into light."  
"Before long the white moon stood in the wood."  
"The land was falling away but the whole path grew low,"  
"Hope I wish to you."  
"For a moment frodo saw no colour , but with a mile , a small sword that lay in leaves falling like stars."  
"Doubt was a day off; but we have better doubt the ruling chance that we should go , so that a year have wandered upon you"  
 yes , there must be many dark creatures, and they have gone tonight.
"At the far brow the hills were rising, and the line of trees lies beyond the earth, pale and brown"
"I do not know, if we ought to go, Elrond answered."
"I was rather glad to return, and a song passed out and to that land of the river."
"And you can say that they are shining, answered Strider, looking at the fire.


These sentences definitely capture the feel of Tolkein's writing, but are all original sentences! I especially love the fact that it creates its own simile, "leaves falling like stars", that does not appear in the original text.

### Preprocessing
The preprocessing stage can be broken down into these steps:
1. Load html file from http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_1__en.htm
2. Parse with BeautifulSoup to get just the text
3. Remove capitalization and some punctuation with Regex
4. Build vocabulary that associates each word to a unique id. This includes word_to_id (dictionary), and id_to_word (direct access table)
5. Map each word in the text to its id
6. Add each word_id in the text to dataset as a label with the 19 word_ids before it as its feature
7. This list of (Feature, Label) pairs is the training data

### Network Architecture
The network has the following architecture:
Input -> Embedding layer -> LSTM layers -> Dropout layer -> Fully Connected Linear Layer -> Output

Input is the 19 preceding word_ids I mentioned before.  
Output is a list of vocab_size values, where a higher value means that that word is more likely to occur next.

### Network Training
Using the network architecture stated above and cross entropy loss, I find the weights that minimize loss on the training data. These weights are calculated with gradient descent using Pytorch's Autograd package.

### Text Generation
By applying the softmax function to the output of the network, I get a probability distribution over all words in the vocabulary. Then, I take a weighted random choice of these to decide the next word.
