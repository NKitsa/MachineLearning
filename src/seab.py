import seaborn as sb
import matplotlib.pyplot as ml
iris_dataset = sb.load_dataset('iris')
print(iris_dataset.head())
# seaborn จะใช้ได้ก็ต่่อเมื่อมีข้อมูลอยู่แล้ว
sb.set()
# จะแสดงเป็น area และการกระจายของข้อมูล โดยเปรียบเทียบข้อมูล
sb.pairplot(iris_dataset,hue= 'species',size=2)
ml.show()