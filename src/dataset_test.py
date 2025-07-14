from sklearn import datasets
import matplotlib.pyplot as plt


# dataset แบ่งได้ 2 แบบ
# 1) training(tr) set เอาไปใช้เพื่อสร้าง Model
# 2) test set(ts) คือข้อมูลที่เอาไปทดสอบถ้าดีเอาไปใช้จริง
# แบ่งได้ dataset = tr 75% ts 25%
# .shape คือดูขนาด
# iris_dataset = datasets.load_iris()
digit_dataset = datasets.load_digits()
# เอา keys  มาดูสำคัญมาก
# print(iris_dataset.keys())

#  index ของภาพ
print(digit_dataset['target'][0])
plt.imshow(digit_dataset['images'][0], cmap=plt.get_cmap('gray'))
plt.show()

print(digit_dataset.keys())
print(digit_dataset['images'][:15])
print(digit_dataset['images'][:15].shape)

# คือข้อมูลที่เก็บสายพันธ์
# print(iris_dataset['target_names'])

# 
# print(iris_dataset['data'][0:10])