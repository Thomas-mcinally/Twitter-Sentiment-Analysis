#Test out Sentiment analysis API with some examples to quality control
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