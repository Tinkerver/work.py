for a in range(0, 30):
    if a % 5 == 0:
        c = 0
        b = a // 5
    else:
        b = a // 5 + 1 - (5 - a % 5)
        c = 5 - a % 5
    print(a, b, c,sep='\t')
