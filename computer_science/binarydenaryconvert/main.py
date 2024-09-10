#Leighton class10b 9/9/2024
'''base2 to base 10'''
bn = input()
dn = 0
for i in range (len(bn)):
    d = int(bn[(len(bn)-1-i)])
    dn += d* (2**i)
print(dn)