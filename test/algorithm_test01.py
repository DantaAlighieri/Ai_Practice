
word = "apple"
letters = 'abcdefghijklmnopqrstuvwxyz'
splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

# inserts = [L + c + R for L, R in splits for c in letters]
# deletes = [L + R[1:] for L, R in splits if R]
# replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
print(splits)
inserts = [L + c + R for L, R in splits for c in letters]
print(inserts)
if "s":
    print("123")
else:
    print("234")