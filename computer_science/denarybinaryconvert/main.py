n = int(input("Enter a denary number: "))
s = ''
if n == 0:
    s = '0'
else:
    while n > 0:
        if n % 2 == 1:
            s = '1' + s
        else:
            s = '0' + s
        n = n // 2
print(s)
