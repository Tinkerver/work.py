# import math
#
# index = 5
# n = 2 ** index
#
# s = [i for i in range(n)]
#
# for i in range(index):
#     print(s)
#     groups = 2 ** i
#     group_len = len(s) / 2 ** i
#     s_temp = [i for i in range(n)]
#     for x in range(n // 2):
#         k = x % groups
#         s_temp[(x - k) * 2 + k] = s[x]
#         s_temp[(x - k) * 2 + k + groups] = s[x + n // 2]
#
#     s = s_temp
# print(s)

import math

index = 5
n = 2 ** index

s = [i for i in range(n)]

for i in range(index - 1):
    print(s)
    groups = 2 ** i
    group_len = len(s) / 2 ** i
    s_temp = [i for i in range(n)]
    for x in range(n // 2):
        k = int(x % (group_len // 4))
        h = x // (group_len // 4)
        j = h % 2
        t = x // (group_len // 2)

        s_temp[int(t * group_len // 4 + k + j * n // 2)] = s[x]
        s_temp[int(t * group_len // 4 + n // 4 + k + j * n // 2)] = s[x + n // 2]
        # s_temp[int((x - k) // 2 + k + j * n // 2)] = s[x]
        # s_temp[int((x - k) // 2 + n // 4 + k + j * n // 2)] = s[x + n // 2]

    s = s_temp
print(s)


import math

index = 5
n = 2 ** index

s = [i for i in range(n)]

for i in range(index-1,-1,-1):
    print(s)
    groups = 2 ** i
    group_len = len(s) / 2 ** i
    s_temp = [i for i in range(n)]
    for x in range(n // 2):
        k = x % groups
        s_temp[(x - k) * 2 + k] = s[x]
        s_temp[(x - k) * 2 + k + groups] = s[x + n // 2]

    s = s_temp
print(s)
