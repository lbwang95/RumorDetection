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

# TODO: replace the file path
# f = open('./source_tweets_id.txt')
f = open("./source_tweets_id_test.txt", mode="r")
tweet_ids = f.readlines()
f.close()

tweet_num = len(tweet_ids)

source_tweets_txt_output = ""

f_graph = open("./twitter_graph.txt", mode="a")

if not os.path.exists("./tree"):
    os.mkdir("./tree")

if not os.path.exists("./retweet_user_tweets"):
    os.mkdir("./retweet_user_tweets")

for index, tweet_id in enumerate(tweet_ids):
    tweet_id = int(tweet_id.strip())

    output_list = ""
    list_root = ["ROOT", "ROOT", "0.0"]

    # try:
    res_source = api.GetStatus(tweet_id, trim_user=False)

    # source_tweet_txt
    tweet_text = res_source.text.lower()  # make all characters lower case
    # replace http(s)~ with URL
    tweet_text = re.sub(
        r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "URL", tweet_text
    )
    tweet_id_text = "{}\t{}\n".format(tweet_id, tweet_text)
    source_tweets_txt_output += tweet_id_text

    source_user_id = res_source.user.id_str
    source_tweet_id = res_source.id_str
    source_tweet_post_datetime = parse(res_source.created_at)

    source_list = [source_user_id, source_tweet_id, "0.0"]

    header = "{}->{}\n".format(list_root, source_list)
    output_list += header

    graph_output_line = "{}\t".format(tweet_id)

    # TODO: change count up to 100
    res = api.GetRetweets(tweet_id, count=10, trim_user=False)

    for retweet in res:
        tweet_id = retweet.id_str
        post_datetime = parse(retweet.created_at)
        user_id = retweet.user.id_str
        user_location = retweet.user.location if retweet.user.location else "none"

        post_time_delay_seconds = (
            post_datetime - source_tweet_post_datetime
        ).total_seconds()
        post_time_delay_minutes = int(post_time_delay_seconds / 60)

        list = [user_id, tweet_id, post_time_delay_minutes]

        list_formatted = "{}->{}".format(source_list, list)
        output_list += list_formatted

        graph_output_line += " {}:{}".format(user_id, post_time_delay_minutes)

        # Gather tweets by the retweet user
        # TODO: cahnge count up to 200
        user_tweets = api.GetUserTimeline(
            user_id, count=30, include_rts=False, exclude_replies=True
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

        tweet_data_filename = "./retweet_user_tweets/{}.txt".format(user_id)
        with open(tweet_data_filename, mode="w") as o:
            o.write(tweet_data)

    output_filename = "./tree/{}.txt".format(source_tweet_id)
    with open(output_filename, mode="w") as o:
        o.write(output_list)

    graph_output_line += "\n"
    f_graph.write(graph_output_line)

    print("\rcompleted tweets: {}/{}".format(index + 1, tweet_num), end="")


output_filename = "source_tweets.txt"
with open(output_filename, mode="w") as o:
    o.write(source_tweets_txt_output)

f_graph.close()
