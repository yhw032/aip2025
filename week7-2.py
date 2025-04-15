class MyClass:
    def __init__(self, num):
        self.numbers1 = [n for n in range(num)]
        self.numbers2 = [n**2 for n in range(num)]

    def __getitem__(self, idx):
        return self.numbers1[idx], self.numbers2[idx]
    
    def __len__(self):
        return len(self.numbers1)

obj = MyClass(10)

for i in range(obj.__len__()):
    print(obj.__getitem__(i))