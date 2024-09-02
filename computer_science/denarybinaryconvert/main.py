'''Denary to Binary convert'''
n = '45'
s=''
while True:
    if n%2==1:
        s=s+'1'
    else:
        s=s+'0'

    n=n//2
print(s)