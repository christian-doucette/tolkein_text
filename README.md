# Tolkein Text
Tolkein Text is a personal project that I created to practice implementing Machine Learning, specifically NLP, in Pytorch. It uses a Neural Network Language Model to generate text based off *The Lord of the Rings*.

### Some Examples
"Before long the white moon stood in the wood."  
"Arrows fell from the sky like lightning hurrying down."  
"At that moment Faramir came in and gazed suddenly into the sweet darkness."  
"Ever the great vale ran down into the darkness. Darker the leaves rolled suddenly through the summer mist."  
"Then they felt once more: as swiftly as they were under the wind of the hills."  
"And the moon was in the dark, and they lay all, grey and fading, terrible and fair."  
"The weather pointed near the trees as the ring had lain."  

These sentences definitely capture the feel of Tolkein's writing, but are all original sentences! I especially love the fact that it creates its own simile, "Arrows fell from the sky like lightning hurrying down," that does not appear in the original text.

### Preprocessing
The preprocessing stage can be broken down into these steps:
1. Load in the full *Lord of the Rings* text
2. Remove capitalization and some punctuation with Regex
3. Build vocabulary that associates each word to a unique id. This includes word_to_id (dictionary), and id_to_word (direct access table)
4. Map each word in the text to its id
5. Add each word_id in the text to dataset as a label with the n word_ids before it as its feature (currently using n=9)
6. This list of (Feature, Label) pairs is the training data

### Network Architecture
The network has the following architecture:
Input -> Embedding layer -> LSTM layers -> Dropout layer -> Fully Connected Linear Layer -> Output

Input is the n preceding word_ids I mentioned before.  
Output is a list of vocab_size values, where a higher value means that that word is more likely to occur next.

### Network Training
Using the network architecture stated above and cross entropy loss, I find the weights that minimize loss on the training data. These weights are calculated with gradient descent using Pytorch's Autograd package.

### Text Generation
By applying the softmax function to the output of the network, I get a probability distribution over all words in the vocabulary. Then, I take a weighted random choice of these to decide the next word.
