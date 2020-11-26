#! /usr/local/bin/python3
import os
import re
import sys
from datetime import *

# import pprint
import twitter
from dateutil.parser import *

CONSUMER_KEY = "pX4oToEI6SAo5PFBRwuBsJjfT"
CONSUMER_SECRET_KEY = "bOn7hhSH0e5EaUIda54iU81v0pJ0cDhuXXq2lZsdyj6B4w99dq"
ACCESS_TOKEN_KEY = "1307406476-gpYVuNsPkV4Nt1bHofe7uRukSoHQucuinS9CDuc"
ACCESS_TOKEN_SECRET_KEY = "u6rX6ZoYCyyhQnECzGrG1MyYIhBjY7MiRrpDkkJHq8iFb"

# pp = pprint.PrettyPrinter(indent=4)

api = twitter.Api(
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET_KEY,
    access_token_key=ACCESS_TOKEN_KEY,
    access_token_secret=ACCESS_TOKEN_SECRET_KEY,
)

if not os.path.exists("./retweet_user_tweets"):
    os.mkdir("./retweet_user_tweets")

f_userID_list = open("./retweet_user_tweets/user_ids.txt", mode="w")

f = open("./dataset/twitter15_graph.tsv", mode="r")
lines = f.readlines()
tweet_num = len(lines)
user_ids = []

for index, line in enumerate(lines):
    # if index < 1:
    source_tweet_id = line.split("\t")[0]
    source_tweet_dirname = "{}/{}".format("retweet_user_tweets", source_tweet_id)
    if not os.path.exists(source_tweet_dirname):
        os.mkdir(source_tweet_dirname)

    dataArray = line.split("\t")[1].split(" ")
    for data in dataArray:
        user_id = data.split(":")[0].strip()
        # user_ids.append(user_id)

        # user_id_output = "{}\n".format(user_id)
        # f_userID_list.write(user_id_output)

        try:
            user_tweets = api.GetUserTimeline(
                user_id, count=3, include_rts=False, exclude_replies=True
            )
            tweet_data = ""
            for tweet in user_tweets:
                # replace http(s)~ with URL
                tweet_text = re.sub(
                    r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)",
                    "URL",
                    tweet.text.lower(),
                )
                posted_at = parse(tweet.created_at)
                tweet_data += "{}\t{}\t{}\n".format(tweet.id_str, tweet_text, posted_at)

            tweet_data_filename = "{}/{}.txt".format(source_tweet_dirname, user_id)
            with open(tweet_data_filename, mode="w") as o:
                o.write(tweet_data)
        except:
            pass
            # print("user not found")

    print("\rcompleted tweets: {}/{}".format(index + 1, tweet_num), end="")

f.close()
