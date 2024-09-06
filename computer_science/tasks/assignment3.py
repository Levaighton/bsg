l = []
while True:
	c = input('data?')
	print(c)
	l.append(c)
	if c == ' ':
		break
print(l[:len(l)-1])
