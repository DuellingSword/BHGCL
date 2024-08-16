import os
from collections import defaultdict

# Step 1: Read data and compute interactions
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
        # Record user and item interactions
        user_interactions[user] += 1
        item_interactions[item] += 1

# Step 2: Filter core users and items
core_users = [user for user, count in user_interactions.items() if count >= 5]
core_items = [item for item, count in item_interactions.items() if count >= 5]

# Step 3: Filter data based on core setting
filtered_user_item_src = []
filtered_user_item_dst = []
for user, item in zip(user_item_src, user_item_dst):
    if user in core_users and item in core_items:
        filtered_user_item_src.append(user)
        filtered_user_item_dst.append(item)

user_item_src = filtered_user_item_src
user_item_dst = filtered_user_item_dst
print("Filtering user-item interactions in Amazon based on ratings greater than 3 and ensuring that both users and items have a core of at least 5 interactions:")

# Step 4: Calculate item degrees and sort items by degree
item_degrees = defaultdict(int)
for item in user_item_dst:
    item_degrees[item] += 1

sorted_items = sorted(item_degrees.items(), key=lambda x: x[1])

# Step 5: Divide items into 5 groups based on degree
num_groups = 5
group_size = len(sorted_items) // num_groups
item_groups = defaultdict(list)

for i, (item, degree) in enumerate(sorted_items):
    group_id = i // group_size
    if group_id >= num_groups:  # 没法整除的，多余的放入最后一组
        group_id = num_groups - 1
    item_groups[group_id].append((item, degree))

# Step 6: Calculate and print the average degree for each group
for group_id in range(num_groups):
    group_items = item_groups[group_id]
    total_degree = sum(degree for _, degree in group_items)
    avg_degree = total_degree / len(group_items) if group_items else 0
    print(f"Group {group_id}: {len(group_items)} items, Average Degree: {avg_degree:.2f}")
    print([item for item, _ in group_items])