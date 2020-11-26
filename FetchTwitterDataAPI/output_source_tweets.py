#! /usr/local/bin/python3
import re
from datetime import *

import twitter
from dateutil.parser import *

# import pprint
# pp = pprint.PrettyPrinter(indent=4)

CONSUMER_KEY = "pX4oToEI6SAo5PFBRwuBsJjfT"
CONSUMER_SECRET_KEY = "bOn7hhSH0e5EaUIda54iU81v0pJ0cDhuXXq2lZsdyj6B4w99dq"
ACCESS_TOKEN_KEY = "1307406476-gpYVuNsPkV4Nt1bHofe7uRukSoHQucuinS9CDuc"
ACCESS_TOKEN_SECRET_KEY = "u6rX6ZoYCyyhQnECzGrG1MyYIhBjY7MiRrpDkkJHq8iFb"

# TODO: specify search terms
term = "COVID-19 vaccine"
# TODO: increate count up to 100
count = 10
OUTPUT_FILENAME = "source_tweets.txt"


def searchTweets(terms, count):
    output = api.GetSearch(term=term, count=count, return_json=True)

    tweetArray = []

    for item in output["statuses"]:
        tweet_id = item["id_str"]
        text = item["text"].replace("\n", "")
        # source_tweet_ids += "{} {}\n".format(tweet_id, text)
        obj = {"tweet_id": tweet_id, "text": text}
        tweetArray.append(obj)

    return tweetArray


api = twitter.Api(
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET_KEY,
    access_token_key=ACCESS_TOKEN_KEY,
    access_token_secret=ACCESS_TOKEN_SECRET_KEY,
)


tweets = searchTweets(term, count)

source_tweets_txt_output = ""

for tweet in tweets:
    tweet_id = int(tweet["tweet_id"].strip())

    # res_source = api.GetStatus(tweet_id, trim_user=False)
    # tweet_text = res_source.text.lower()  # make all characters lower case
    tweet_text = tweet["text"]
    # replace http(s)~ with URL
    tweet_text = re.sub(
        r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "URL", tweet_text
    )
    tweet_id_text = "{}\t{}\n".format(tweet_id, tweet_text)
    source_tweets_txt_output += tweet_id_text

with open(OUTPUT_FILENAME, mode="w") as o:
    o.write(source_tweets_txt_output)
