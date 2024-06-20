import tkinter as tk
from tkinter import *


def GJ(t, k):       # 估价函数 h(n) = M + C - k * B
    return t[0] + t[1] - k * t[2]

def create(array, m, c, b, n):      # 判断状态是否合法
    p = array[:]
    if c >= 0 and m >= 0 and m <= n and c <= n and (m >= c or m == 0) and (n - m >= n - c or n - m == 0):       # 判断是否数字合法以及传道士是否会被野人吃掉
        p.insert(0, [m, c, 1 - b])
        for i in open:              # 判断此状态是否以及经过，经过则不需要重复添加
            if p[0] == i[0]:
                return False
        for j in closed:
            if p[0] == j[0]:
                return False
        open.append(p)
        return True
    else:
        return False

def doJob():
    global n, k, closed, open
    n = int(e1.get())
    k = int(e2.get())
# n = int(input("传教士和野人的人数（默认相同）:"))
# k = int(input("船的最大容量："))
    open = []                   # open格式： list[list[list[int]]],open[i]表示第i条路径，open[i][j]表示第i条路经的第j个状态(节点)，存储扩展的节点路径
    closed = []                 # closed格式和open相同，存储已经搜索过的节点路径
    sample = [n, n, 1]
    goal = [0, 0, 0]

    open.append([sample])
    createPoint = searchPoint = 0

    text.delete('1.0', 'end')
    while(True):
        if sample == goal:
            # print("初始状态为目标状态！")
            text.insert('end', "初始状态为目标状态！")
            break
        if len(open) == 0:
            # print("未搜索到解！")
            text.insert('end', "未搜索到解！")
            break
        else:
            t = open.pop(0)
            closed.append(t)

            if t[0] == goal:
                # print("搜索成功！")
                # print("共生成节点数：{}， 共搜索节点数：{}".format(createPoint, searchPoint + 1))
                # print("过河方案如下：")
                # print("[M, C, B]")
                content = "搜索成功！\n共生成节点数：{}，共搜索节点数：{}\n".format(createPoint, searchPoint + 1)
                text.insert('end', content)
                text.insert('end', "过河方案如下：\n[M, C, B]\n")
                for i in t[::-1]:
                    text.insert('end', '     |\n')
                    text.insert('end', str(i) + '\n')
                    # print(i)
                # exit()
                return

            searchPoint += 1
            if t[0][2] == 1:                        # 船在左岸时
                for i in range(1, k + 1):                               # 全部运送传道士过去
                    if create(t, t[0][0] - i, t[0][1], t[0][2], n):
                        createPoint += 1
                    if create(t, t[0][0], t[0][1] - i, t[0][2], n):     # 全部运送野人过去
                        createPoint += 1
                    for r in range(1, k - i + 1):                       # 传道士和野人混合
                        if create(t, t[0][0] - i, t[0][1] - r, t[0][2], n):
                            createPoint += 1
            else:                                   # 船在右岸时
                for i in range(1, k + 1):
                    if create(t, t[0][0] + i, t[0][1], t[0][2], n):
                        createPoint += 1
                    if create(t, t[0][0], t[0][1] + i, t[0][2], n):
                        createPoint += 1
                    for r in range(1, k - i + 1):
                        if create(t, t[0][0] + i, t[0][1] + r, t[0][2], n):
                            createPoint += 1

            for x in range(0, len(open) - 1):           # 重排open表
                m = x
                for y in range(x + 1, len(open)):
                    if GJ(open[m][0], k) > GJ(open[y][0], k):
                        m = y
                if m != x:
                    open[x], open[m] = open[m], open[x]


window = Tk()
window.title('修道士和野人过河方案')
window.geometry('700x520')

# var = StringVar()
l1 = Label(window, text='输入传道士或野人数(默认相等):')
l1.place(x=0, y=0)
l2 = Label(window, text='输入单只船的最大容量:')
l2.place(x=30, y=30)
text = Text(window, height=400, width=200, font=('Verdana', 10))
text.place(x=0, y=60)

e1 = Entry(window, width=50)
e1.place(x=180, y=0)
e2 = Entry(window, width=50)
e2.place(x=180, y=30)
def on_enter(event):
    event.widget.config(bg='#D3D3D3')

def on_leave(event):
    event.widget.config(bg='#F0F0F0')

b1 = Button(window, text='Start', width=6, height=1, relief='flat', command=doJob)
b1.place(x=580, y=0)
b1.bind("<Enter>", on_enter)
b1.bind("<Leave>", on_leave)

b2 = Button(window, text='Quit', width=6, height=1, relief='flat', command=window.quit)
b2.place(x=580, y=30)
b2.bind("<Enter>", on_enter)
b2.bind("<Leave>", on_leave)

window.mainloop()