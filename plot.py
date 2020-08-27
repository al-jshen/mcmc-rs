import matplotlib.pyplot as plt

data = []
with open("data.txt", "r") as f1:
    datalen = int(next(f1))
    print(datalen)
    data = [float(next(f1).strip()) for _ in range(datalen)]
    samples = [float(next(f1).strip()) for _ in f1]

# print(data)
# print(samples)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(samples)
ax2 = fig.add_subplot(122)
ax2.hist(data, bins=50, alpha=0.5)
ax2.hist(samples[int(len(samples) / 2) :], bins=50, alpha=0.5)
plt.show()
