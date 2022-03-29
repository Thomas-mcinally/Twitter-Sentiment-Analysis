#Example sentences and their sentiment
'''
SIMPLE CASES
Coronet has the best lines of all day cruisers.
Bertram has a deep V hull and runs easily through seas.
Pastel-colored 1980s day cruisers from Florida are ugly.
I dislike old cabin cruisers.

ADVANCED CASES
I do not dislike cabin cruisers. (Negation handling)
Disliking watercraft is not really my thing. (Negation, inverted word order)
Sometimes I really hate RIBs. (Adverbial modifies the sentiment)
I'd really truly love going out in this weather! (Possibly sarcastic)
Chris Craft is better looking than Limestone. (Two brand names, identifying the target of attitude is difficult).
Chris Craft is better looking than Limestone, but Limestone projects seaworthiness and reliability. (Two attitudes, two brand names).
The movie is surprising with plenty of unsettling plot twists. (Negative term used in a positive sense in certain domains).
You should see their decadent dessert menu. (Attitudinal term has shifted polarity recently in certain domains)
I love my mobile but would not recommend it to any of my colleagues. (Qualified positive sentiment, difficult to categorise)
Next week's gig will be right koide9! ("Quoi de neuf?", French for "what's new?". Newly minted terms can be highly attitudinal but volatile in polarity and often out of known vocabulary.)
'''


#COMMON SENTIMENT ANALYSIS LIBRARIES
'''
Rule-based models (No ML)
- vader - Optimised for social-media
- textblob

ML-Based models
flair - Pretrained model is trained on IMDB dataset
transformers (BERT - Hugging face) - Multi-lingual sentiment analysis tool optimised for product reviews

Check out this video for how they compare: https://www.youtube.com/watch?v=q0V_z_I9bWU
Good article: https://neptune.ai/blog/sentiment-analysis-python-textblob-vs-vader-vs-flair

Key points:
- Transformers and Flair have significantly longer processing time - dont use if time sensitive
- Different models are trained on different data - the best model to use depends on the text you are studying

'''



#Explain twitter API query choice
'''
Keyword 'wordle'
    - Look for tweets that contain the word 'wordle' in the body of the tweet.

Geolocation UK (to make results easier to interpret, less factors to consider)
    place_country:GB

Retweets not included
    -Can't use place_country for retweets, since place is attached to original tweet

Don't include tweets with media (images/videos/gifs)
    - A lot of images in tweets are memes, and have a significant impact on the sentiment of the tweet
    - Even if the tweet also contains text, the image might totall change the sentiment of the tweet 
    - Example tweet: https://t.co/jD5IgohHni

Don't include advertising tweets
    -Looking for public sentiment, not how advertisers feel 

Language English 
    - VADER only works for english

    query = 'wordle place_country:GB -is:retweet -has:media -is:nullcast lang:en'
'''