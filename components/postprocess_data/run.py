from argparse import ArgumentParser
import logging

import pandas as pd

from component_utils.general import create_artifact_folder

logger = logging.getLogger()

def go(args):
    artifact_path = create_artifact_folder(__file__)

    raw_data = pd.read_csv(args.input_path, sep = "\t", index_col = 0)
    raw_data.drop(columns = ["Gender", "Account Type"], inplace = True)
    raw_data.rename(columns = {"Thread Entry Type": "Tweet Type"}, inplace = True)

    demo = pd.read_csv(args.demo_path)

    news = pd.read_html("https://memeburn.com/2010/09/the-100-most-influential-news-media-twitter-accounts/", header = 0)[0]
    news["@name"] = news["@name"].str.replace("@", "").str.lower()
    news_users = {u.lower(): "news" for u in news["@name"]}

    brands_df = pd.read_excel("../../data/brandfulllist.xlsx", sheet_name = "All 1558", usecols = ["Twitter Handle"])
    brands = (
        brands_df[brands_df["Twitter Handle"] != "NOT AVAILABLE"]
        ["Twitter Handle"]
        .str.replace("@", "")
        .str.lower()
    )
    brands_map = {k: "business" for k in brands.values}

    companies_df = pd.read_excel("../../data/companyfulllist.xlsx", usecols = ["TwitterHandle", "TwitterHandle2"])
    companies = (
        pd.concat([companies_df["TwitterHandle"], companies_df["TwitterHandle2"]], axis = 0)
        .dropna()
        .str.replace("@", "")
        .str.lower()
    )
    companies_map = {k: "business" for k in companies.values}

    demo["Account Type"] = (
        demo["followers_count"].apply(lambda x: "influencer" if x > args.influencer_thresh else "core")
        .mask(demo["Account Type"] != "individual")
        .combine_first(demo["Account Type"])
    )

    demo["Account Type"] = demo["screen"].str.lower().map(brands_map).combine_first(demo["Account Type"])
    demo["Account Type"] = demo["screen"].str.lower().map(companies_map).combine_first(demo["Account Type"])
    demo["Account Type"] = demo["screen"].str.lower().map(news_users).combine_first(demo["Account Type"])
    demo["Gender"] = demo["Gender"].mask(~demo["Account Type"].isin(["core", "influencer"]))

    data = raw_data.merge(right = demo, how = "left", left_on = "Author", right_on = "screen")
    data = data.merge(right = demo, how = "left", left_on = "Thread Author", right_on = "screen", suffixes = ("", "_originator"))
    
    data.to_csv(artifact_path / "metoo_data.csv", sep = "\t")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", type = str, help = "Run mode (local or remote)")
    parser.add_argument("input_path", type = str, help = "Path to data")
    parser.add_argument("demo_path", type = str, help = "Path to inferred author demographics")
    parser.add_argument("influencer_thresh", type = int, help = "follower count threshold for influencer designation")
    args = parser.parse_args()

    go(args)