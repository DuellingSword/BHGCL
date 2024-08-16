import random

remove_percent = 0.30
file_name = 'train.txt'
new_filename = '0.3noise_train.txt'
add_noise_percent = 0.3


def find_max_values(filename):
    first_col_max = 0
    all_nums_max = float('-inf')
    user_items = {}  # 用于存储每个用户的正样本
    with open(filename, 'r') as f:
        for line in f:
            nums = line.strip().split()
            first_num = int(nums[0])
            if first_num > first_col_max:
                first_col_max = first_num
            for num in nums[1:]:
                if float(num) > all_nums_max:
                    all_nums_max = float(num)
            items = nums[1:]
            user_id = int(nums[0])
            if user_id not in user_items:
                user_items[user_id] = set()
            user_items[user_id].update(items)  # 更新用户的正样本
    return first_col_max, all_nums_max, user_items


first_col_max, all_nums_max, user_items = find_max_values(file_name)


# 移除 30% 的边
def remove_edges(user_items, remove_percent):
    new_user_items = {}
    for user, items in user_items.items():
        num_items = len(items)
        num_to_remove = int(num_items * remove_percent)
        remaining_items = set(random.sample(items, num_items - num_to_remove))
        new_user_items[user] = remaining_items
    return new_user_items


user_items = remove_edges(user_items, remove_percent)


def add_noise_edges(user_items, add_noise_percent, all_nums_max):
    new_user_items = {}
    for user, items in user_items.items():
        num_items = len(items)
        num_to_add = int(num_items * add_noise_percent)
        new_items = set(items)
        while len(new_items) < num_items + num_to_add:
            new_item = str(random.randint(0, int(all_nums_max)))
            if new_item not in new_items:
                new_items.add(new_item)
        new_user_items[user] = new_items
    return new_user_items


user_items = add_noise_edges(user_items, add_noise_percent, all_nums_max)

# 将结果写入新的文件
with open(new_filename, 'w') as f_out:
    for user, items in user_items.items():
        items_sorted = sorted(items, key=int)
        f_out.write(f"{user} {' '.join(map(str, items_sorted))}\n")

# 读取文件
with open(new_filename, 'r') as f:
    lines = f.readlines()

# 对每一行进行处理
for i in range(len(lines)):
    # 将当前行的数字拆分成一个列表
    nums = lines[i].strip().split()[1:]
    # 将数字列表中的元素转换成整数类型
    nums = [int(num) for num in nums]
    # 对数字列表进行从小到大排序
    nums = sorted(nums, reverse=False)
    print(nums)
    exit()
    # 将排好序的数字列表转换成字符串
    nums_str = ' '.join([str(num) for num in nums])
    # 将当前行的数字替换为排好序的数字字符串
    lines[i] = lines[i].split()[0] + ' ' + nums_str + '\n'

# 将处理后的行写入新的文件
with open(new_filename, 'w') as f:
    f.writelines(lines)
