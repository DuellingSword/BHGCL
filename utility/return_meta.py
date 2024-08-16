def return_meta(dataset):
    if dataset == "Movielens":
        meta_paths = {
            "user": [["um", "mu"]],
            # "user": [["ua", "au"]],
            # , ["ua", "au"], ["uo", "ou"], ["mu", "ua", "au", "um"]
            # "user": [["um", "mu"], ["um", "mg", "gm", "mu"]],
            "movie": [["mu", "um"], ["mg", "gm"]],
        }
        user_key = "user"
        item_key = "movie"
        ui_relation = "um"
    elif dataset == "Amazon":
        meta_paths = {
            "user": [["ui", "iu"]],
            "item": [["iu", "ui"], ["ic", "ci"], ["ib", "bi"], ["iv", "vi"]],
        }
        user_key = "user"
        item_key = "item"
        ui_relation = "ui"
    elif dataset == "DoubanBook":
        meta_paths = {
            "user": [["ub", "bu"], ["ug", "gu"]],
            "book": [["bu", "ub"], ["ba", "ab"]],
            # "user": [["ub", "bu"], ["ug", "gu"], ["uu1", "uu2"], ["ul", "lu"]],
            # "book": [["bu", "ub"], ["ba", "ab"], ["bp", "pb"], ["by", "yb"]],
        }
        user_key = "user"
        item_key = "book"
        ui_relation = "ub"
    elif dataset == "DoubanMovie":
        meta_paths = {
            "user": [["um", "mu"], ["ug", "gu"], ["uu1", "uu2"]],
            "movie": [["mu", "um"], ["mt", "tm"]],
            # "user": [["um", "mu"], ["ug", "gu"], ["uu1", "uu2"]],
            # "movie": [["mu", "um"], ["ma", "am"], ["md", "dm"], ["mt", "tm"]],
        }
        user_key = "user"
        item_key = "movie"
        ui_relation = "um"
    elif dataset == "Yelp":
        meta_paths = {
            # "user": [["ub", "bu"], ["uu1", "uu2"], ["uc", "cu"]],
            # "business": [["bu", "ub"], ["bc", "cb"], ["bc1", "cb1"]],
            "user": [["ub", "bu"]],
            "business": [["bu", "ub"], ["bc", "cb"], ["bc1", "cb1"]],
        }
        user_key = "user"
        item_key = "business"
        ui_relation = "ub"
    else:
        print("Available datasets: Movielens, Amazon, DoubanBook, DoubanMovie, Yelp.")
        raise NotImplementedError

    return meta_paths, user_key, item_key, ui_relation
