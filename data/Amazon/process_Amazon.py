import os
import dgl
import random
import pickle as pkl
from collections import defaultdict



# user_item
user_item_src = []
user_item_dst = []
user_interactions = defaultdict(int)
item_interactions = defaultdict(int)
with open(os.path.join("user_item.dat")) as fin:
    for line in fin.readlines():
        _line = line.strip().split("\t")
        user, item, rate = int(_line[0]), int(_line[1]), int(_line[2])
        if rate > 3:
            user_item_src.append(user)
            user_item_dst.append(item)

        # 记录user和item的交互次数
        user_interactions[user] += 1
        item_interactions[item] += 1

# 过滤出满足条件的user和item
core_users = [user for user, count in user_interactions.items() if count >= 5]
core_items = [item for item, count in item_interactions.items() if count >= 5]

# 过滤数据，只保留符合“core setting”的记录
filtered_user_item_src = []
filtered_user_item_dst = []

for user, item in zip(user_item_src, user_item_dst):
    if user in core_users and item in core_items:
        filtered_user_item_src.append(user)
        filtered_user_item_dst.append(item)

user_item_src = filtered_user_item_src
user_item_dst = filtered_user_item_dst
print(
    "Filtering user-item interactions in Amazon based on ratings greater than 3 and ensuring that both users "
    "and items have a core of at least 5 interactions:")

# 将用户和物品的配对重新组合成一个列表，每个元素是一个(user, item)对
user_item_pairs = list(zip(user_item_src, user_item_dst))

# 为了确保数据集分割的随机性，我们先随机打乱这个列表
random.seed(42)  # 设置随机种子以保证结果的可重复性
random.shuffle(user_item_pairs)

# 计算训练数据集的大小（80%）
train_size = int(0.8 * len(user_item_pairs))

# 分割数据集为训练集和测试集
train_pairs = user_item_pairs[:train_size]
test_pairs = user_item_pairs[train_size:]

# 创建一个字典，用于存储每个用户的物品集合
train_dict = defaultdict(set)
test_dict = defaultdict(set)

# 填充训练集字典
for user, item in train_pairs:
    train_dict[user].add(item)

# 填充测试集字典
for user, item in test_pairs:
    test_dict[user].add(item)


# 函数用于将字典数据写入文件
def write_to_file(file_path, data_dict):
    with open(file_path, 'w') as file:
        for user in sorted(data_dict.keys()):
            items = sorted(data_dict[user])
            # 将用户ID和物品ID列表转换成字符串格式，物品ID之间用空格分隔
            items_str = ' '.join(map(str, items))
            # 写入一行数据：用户和其对应的物品列表
            file.write(f"{user} {items_str}\n")


# 写入训练数据到 train.txt
write_to_file("train.txt", train_dict)

# 写入测试数据到 test.txt
write_to_file("test.txt", test_dict)

print("Files 'train.txt' and 'test.txt' have been created.")

# build graph
train_user_src, train_user_dst = zip(*train_pairs)
# item_view
item_view_src = []
item_view_dst = []
with open(os.path.join("item_view.dat")) as fin:
    for line in fin.readlines():
        _line = line.strip().split(",")
        item, view = int(_line[0]), int(_line[1])
        item_view_src.append(item)
        item_view_dst.append(view)

# item_category
item_category_src = []
item_category_dst = []
with open(os.path.join("item_category.dat")) as fin:
    for line in fin.readlines():
        _line = line.strip().split(",")
        item, category = int(_line[0]), int(_line[1])
        item_category_src.append(item)
        item_category_dst.append(category)

# item_brand
item_brand_src = []
item_brand_dst = []
with open(os.path.join("item_brand.dat")) as fin:
    for line in fin.readlines():
        _line = line.strip().split(",")
        item, brand = int(_line[0]), int(_line[1])
        item_brand_src.append(item)
        item_brand_dst.append(brand)
hg = dgl.heterograph(
    {
        ("user", "ui", "item"): (train_user_src, train_user_dst),
        ("item", "iu", "user"): (train_user_dst, train_user_src),
        ("item", "iv", "view"): (item_view_src, item_view_dst),
        ("view", "vi", "item"): (item_view_dst, item_view_src),
        ("item", "ic", "category"): (item_category_src, item_category_dst),
        ("category", "ci", "item"): (item_category_dst, item_category_src),
        ("item", "ib", "brand"): (item_brand_src, item_brand_dst),
        ("brand", "bi", "item"): (item_brand_dst, item_brand_src),
    }
)
print("Graph constructed.")
with open(os.path.join("Amazon_hg.pkl"), "wb") as file:
    pkl.dump(hg, file)
