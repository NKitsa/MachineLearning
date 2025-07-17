# LinearRegrerss คือการถดถอยเชิงเส้น x= ตัวที่ทราบค่า y= ตัวไม่ืทราบค่า ***ถ้าเรารู้ค่า x เราก็จะรู้ค่า y
# เป็นการหาคสพระหว่าง2 ตัวแปร หาตัวแปรที่ไม่ทราบค่าจากตัวแปรที่เราทราบค่า y = ax+b
# แบ่งได้ 2 รูปแบบ 1) ถ้ามีทิศทางเดียวกัน ถ้าค่า x เพิ่มขึ้น y ก็เพิ่มขึ้น 2) แปรผกผันตัวนึงเพิ่มตัวนึงลดลง 3) การกระจายข้อมูล
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# linspac คือสร้างค่าที่อยู่ในช่วง
x_scatter = np.random
# การจำลองข้อมูลทดสอบ
info_x = x_scatter.rand(50)*10
y = 2*info_x+x_scatter.rand(50)

model = LinearRegression()
# print(info_x)
twod_x = info_x.reshape(-1,1)
# print(twod_x)
# trian เวลาเทรนต้องใช้ข้อมูล 2 มิติ
model.fit(twod_x,y)
# r-squre R-squared (R²) หรือที่เรียกว่า ค่าสัมประสิทธิ์การตัดสินใจ (coefficient of determination) เป็นค่าทางสถิติที่ใช้วัดว่า โมเดลการทำนายสามารถอธิบายความแปรปรวนของข้อมูลได้ดีแค่ไหน
# โดยค่าจะมี 0-100 ถ้า 0.97->0.97*100 = 97% โดยค่า R² ไม่สามารถบอกว่าโมเดลดีแน่ ๆ ได้เสมอ ควรดูค่าประเมินอื่น ๆ เช่น MAE, RMSE ด้วย
print(model.score(twod_x,y))

# test model
xfit = np.linspace(-1,11)
xset = xfit.reshape(-1,1)

# ผลลัพธ์กาพยากรณ์
yfit = model.predict(xset)

plt.scatter(yfit,xset)
plt.show()







# x = np.linspace(-5,5,100)
# y = 2*x+1
# # scatter กระจาย
# plt.scatter(x,y)
# plt.plot(x,y,'-r',label = 'y=2x+1')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc="upper left")
# plt.title("y=2x+1")
# plt.grid()
# plt.show()