# 消去连接词
import re
from tkinter import Tk, messagebox
from tkinter import *

def delInclude(clauses):
    right_bracket = 1

    index = clauses.find('>')
    if index == -1:
        return clauses
    temp = list(clauses)
    temp[index] = '!'

    index -= 2
    while right_bracket and index >= 0:
        if temp[index] == ')':
            right_bracket += 1
        if temp[index] == '(':
            right_bracket -= 1
        index -= 1

    if temp[index].isalpha() and temp[index - 1] == ')':
        temp.insert(index - 4, '~')
    elif temp[index].isalpha():
        temp.insert(index, '~')
    elif temp[index - 2] == '@' or temp[index - 2] == '#':
        temp.insert(index - 3, '~')
    else:
        temp.insert(index + 1, '~')

    return "".join(temp)

# 减少否定符号的辖域
def decNegRand(clauses):
    left_bracket = 0

    index = clauses.find('~(')
    if index == -1:
        return clauses

    temp = list(clauses)
    temp.pop(index)
    left_bracket += 1
    index += 1
    while left_bracket:
        if temp[index] == '(':
            left_bracket += 1
        elif temp[index] == ')':
            left_bracket -= 1
        elif temp[index] == '@':
            temp[index] = '#'
        elif temp[index] == '#':
            temp[index] = '@'
        elif temp[index] == '~':
            temp.pop(index)
        elif temp[index] == '!':
            temp[index] = '%'
        elif temp[index] == '%':
            temp[index] = '!'
        elif temp[index].isalpha() and temp[index + 1] == '(':
            temp.insert(index, '~')
            index += 1
        index += 1

    if temp[index] == '(':
        temp.insert(index, '~')

    return "".join(temp)

# 变元标准化
def standardVar(clauses):
    i = 0
    left_bracket = 1
    index1 = clauses.find('@')
    if index1 == -1:
        index1 = 100
    index2 = clauses.find('#')
    if index2 == -1:
        index2 = 100
    index = min(index1, index2)
    temp = list(clauses)
    var[temp[index + 1]] = 1

    index += 4
    while index < len(clauses) and left_bracket:
        if temp[index] == '(':
            left_bracket += 1
        elif temp[index] == ')':
            left_bracket -= 1
        elif temp[index] == '@' or temp[index] == '#':
            old = ''
            new = ''
            if var[temp[index + 1]] == 1:
                old = temp[index + 1]
                temp[index + 1] = [k for k, v in var.items() if k not in temp and var[k] == 0][0]
                var[temp[index + 1]] = 1
                new = temp[index + 1]
            l = index + 3
            r = l + 1
            left_bracket1 = 1
            while left_bracket1:
                if temp[r] == '(':
                    left_bracket1 += 1
                elif temp[r] == ')':
                    left_bracket1 -= 1
                elif temp[r] == old:
                    temp[r] = new
                r += 1
            l -= 4
            s = "".join(temp[l:r])
            temp = temp[:l] + list(standardVar(s)) + temp[r:]
        index += 1

    return "".join(temp)

# 化为前束范式
def convertToFront(clauses):
    words = []

    temp = list(clauses)
    index = clauses.find('@')
    while index != -1:
        words.extend(temp[index - 1:index + 3])
        # print(f"words={words}")
        for i in range(index - 1, index + 3):
            temp.pop(index - 1)

        clauses = "".join(temp)
        temp = list(clauses)
        index = clauses.find('@')

    index = clauses.find('#')
    while index != -1:
        words.extend(temp[index - 1:index + 3])
        for i in range(index - 1, index + 3):
            temp.pop(index - 1)
        clauses = "".join(temp)
        temp = list(clauses)
        index = clauses.find('#')
    words.extend(temp)

    return "".join(words)

# 消去存在量词
def delExists(clauses):
    new = []
    res = []        # 存放下标
    pat1 = '@'

    for each in re.finditer(pat1, clauses):
        res.append(each.span()[0])

    index = clauses.find('#')
    temp = list(clauses)
    old = temp[index + 1]
    for i in res:
        new.append(temp[i + 1])
        new.append(',')
    new.pop()

    while index != -1:
        for i in range(index - 1, index + 3):
            temp.pop(index - 1)

        j = 0
        while True:
            if temp[j] == old:
                temp.pop(j)
                temp = temp[:j] + ['g', '('] + new + [')'] + temp[j:]
            j += 1
            if j >= len(temp):
                break
        clauses = "".join(temp)
        index = clauses.find('#')
        old = temp[index + 1]
        temp = list(clauses)

    return clauses

def seperate(clause):
    words = dict()
    word = []
    con = []
    temp = clause
    i = 1
    j = 0
    left_bracket = 0

    while i < len(clause) - 1:
        word.append(temp[i])
        if temp[i] == '(':
            left_bracket += 1
        elif temp[i] == ')':
            left_bracket -= 1
        if left_bracket == 0 and temp[i] == ')':
            words[j] = word
            j += 1
            word = []
            con.append(temp[i + 1])     # 连接符号
            i += 1
        i += 1
    if len(con) and con[-1] != '!' and con[-1] != '%':
        con.pop()

    return words, con

# 化为skolem标准型
def convertToAnd(clauses):
    left_bracket = 0
    j = 0
    words = dict()
    temp = list(clauses)
    res = []

    var_num = temp.count('@')
    i = 4 * var_num + 1
    word = []
    con = []
    while i < len(clauses) - 1:
        word.append(temp[i])
        if temp[i] == '(':
            left_bracket += 1
        elif temp[i] == ')':
            left_bracket -= 1
        if left_bracket == 0 and temp[i] == ')':
            words[j] = word
            # print(f"word={word}")
            j += 1
            word = []
            con.append(temp[i + 1])     # 连接符号
            i += 1
        i += 1
    if len(con) and con[-1] != '!' and con[-1] != '%':
        con.pop()
    # print(f"words={words}")
    # print(f"connection={con}")
    while True:
        if '!' not in con:
            break
        index = con.index('!')          # 找到第一个析取符号下标
        w1, c1 = seperate(words[index])
        w2, c2 = seperate(words[index + 1])
        # print(f"c1={c1}\nw1={w1}\n")
        # print(f"c2={c2}\nw2={w2}\n")
        if len(c1) and '!' not in c1 and ('!' in c2 or len(c2) == 0):     # 得到的第一个子句集连接方式都是合取
            for i in range(len(w1)):
                res.extend(['('] + words[index + 1] + ['!'] + w1[i] + [')'])
                res.append('%')
            for i in range(index):
                res.extend(words[i])
                res.append('%')
            for i in range(index + 2, len(words)):
                res.extend(words[i])
                res.append('%')
            res.pop()
            con.pop()
        elif len(c1) and len(c2) and '!' in c1 and '!' in c2 or (len(c1) == 0 and '!' in c2) or (len(c2) == 0 and '!' in c1):
            i = 0
            temp1 = []
            temp2 = []
            while i < len(c1):
                temp1 = temp1 + w1[i] + list(c1[i])
                i += 1
            if len(temp1):
                temp1 = ['('] + temp1 + w1[i] + [')']
                # print(f"temp1={temp1}")
                new_clause1 = convertToAnd("".join(temp1))
                words[index] = ['('] + list(new_clause1) + [')']
            i = 0
            while i < len(c2):
                temp2 = temp2 + w2[i] + list(c2[i])
                i += 1
            if len(temp2):
                temp2 = ['('] + temp2 + w2[i] + [')']
                new_clause2 = convertToAnd("".join(temp2))
                words[index + 1] = ['('] + list(new_clause2) + [')']
        # elif len(c2) and '!' not in c2 and ('!' in c1 or len(c1) == 0):
        else:
            for i in range(len(w2)):
                res.extend(['('] + words[index] + ['!'] + w2[i] + [')'])
                res.append('%')
            for i in range(index):
                res.extend(words[i])
                res.append('%')
            for i in range(index + 2, len(words)):
                res.extend(words[i])
                res.append('%')
            res.pop()
            con.pop()

    # print(f"res={res}")
    clauses = "".join(res)

    return clauses

# 消去合取词
def delAnd(clauses):
    temp = list(clauses)
    words = []
    word = []
    left_bracket = 0
    i = 0

    while i < len(clauses):
        word.append(temp[i])
        if temp[i] == '(':
            left_bracket += 1
        elif temp[i] == ')':
            left_bracket -= 1
        if left_bracket == 0 and temp[i] == ')':
            word.pop(0)
            word.pop()
            words.append("".join(word))
            word = []
            i += 1
        i += 1

    return words

# 子句变量标准化
def changeName(clauses_list):
    cnt = 1
    for i in clauses_list:
        ctemp = temp = list(i)
        j = 0
        for each in temp:
            if each in var.keys() and var[each]:
                ctemp.insert(j + 1, str(cnt))
            j += 1
        clauses_list[cnt - 1] = "".join(ctemp)
        cnt += 1

    return clauses_list

# (@x)(P(x)>((@y)(P(y)>P(f(x,y)))%~(@y)(Q(x,y)>P(y))))  (@x)((@y)P(x,y)>~(@y)(Q(x,y)>R(x,y)))
var = {'x': 0, 'y': 0, 'z': 0, 'm': 0, 'n': 0, 'k': 0, 'r': 0}


def step1():
    # clauses = input("输入需要转换的谓词公式:")
    # 1
    global clauses
    clauses = e.get()
    text.delete('1.0', 'end')
    res = ""
    while True:
        res = delInclude(clauses)
        if res == clauses:
            break
        clauses = res
    print(f"消去连接词:\n{clauses}")
    text.insert('end', "\n\n{}\n\n".format(clauses))


def step2():
    # 2
    global clauses
    while True:
        res = decNegRand(clauses)
        if res == clauses:
            break
        clauses = res

    print(f"减少否定符号的辖域：\n{clauses}")
    text.insert('end', "\n\n{}\n\n".format(clauses))


def step3():
    # 3
    global clauses
    clauses = standardVar(clauses)
    print(f"变元标准化：\n{clauses}")
    text.insert('end', "\n\n{}\n\n".format(clauses))


def step4():
    # 4
    global clauses
    clauses = convertToFront(clauses)
    print(f"化为前束范式：\n{clauses}")
    text.insert('end', "\n\n{}\n\n".format(clauses))


def step5():
    # 5
    global clauses
    clauses = delExists(clauses)
    print(f"消去存在量词:\n{clauses}")
    text.insert('end', "\n\n{}\n\n".format(clauses))


def step67():
    # 6、7
    global clauses
    clauses = convertToAnd(clauses)
    print(f"skolem标准化并略去全称量词：\n{clauses}")
    text.insert('end', "\n\n{}\n\n".format(clauses))


def step8():
    # 8
    global clauses
    global clauses_list
    clauses_list = delAnd(clauses)
    print(f"消去合取词：\n{clauses_list}")
    text.insert('end', "\n\n{}\n\n".format(clauses_list))


def step9():
    # 9
    global clauses_list
    clauses_list = changeName(clauses_list)
    print(f"子句变元标准化：\n{clauses_list}")
    text.insert('end', "\n\n{}\n\n".format(clauses_list))


window = Tk()
window.title("九步法消解子句集")
window.geometry('800x600')

l = Label(window, text='请输入需要转换的谓词公式：')
l.place(x=0, y=0)

e = Entry(window, width=80)
e.place(x=152, y=0)

def on_enter(event):
    event.widget.config(bg='#F0F0F0')

def on_leave(event):
    event.widget.config(bg='white')

def on_enter1(event):
    event.widget.config(bg='#D3D3D3')

def on_leave1(event):
    event.widget.config(bg='#F0F0F0')

def checkHelp():
    messagebox.showinfo('帮助', '本例程规定输入时蕴含符号为>，全称量词为@，存在量词为#，取反为~，析取为！，合取为%，函数名请使用一个字母\n')

b0 = Button(window, text='退出', relief='flat', command=window.quit)
b0.place(x=750, y=0)
b0.bind("<Enter>", on_enter1)
b0.bind("<Leave>", on_leave1)

text = Text(window, height=300, width=100, font=('Verdana', 10))
text.place(x=0, y=30)

help = Button(window, text='查看帮助', relief='flat', command=checkHelp, bg='white')
help.place(x=740, y=550)
help.bind("<Enter>", on_enter)
help.bind("<Leave>", on_leave)

step1 = Button(window, text='消去连接词', relief='flat', command=step1, bg='white')
step1.place(x=2, y=32)
step1.bind("<Enter>", on_enter)
step1.bind("<Leave>", on_leave)

step2 = Button(window, text='减少否定符号辖域', relief='flat', command=step2, bg='white')
step2.place(x=2, y=95)
step2.bind("<Enter>", on_enter)
step2.bind("<Leave>", on_leave)

step3 = Button(window, text='变元标准化', relief='flat', command=step3, bg='white')
step3.place(x=2, y=160)
step3.bind("<Enter>", on_enter)
step3.bind("<Leave>", on_leave)

step4 = Button(window, text='化为前束范式', relief='flat', command=step4, bg='white')
step4.place(x=2, y=225)
step4.bind("<Enter>", on_enter)
step4.bind("<Leave>", on_leave)

step5 = Button(window, text='消去存在量词', relief='flat', command=step5, bg='white')
step5.place(x=2, y=290)
step5.bind("<Enter>", on_enter)
step5.bind("<Leave>", on_leave)

step67 = Button(window, text='skolem标准化并略去全称量词', relief='flat', command=step67, bg='white')
step67.place(x=2, y=355)
step67.bind("<Enter>", on_enter)
step67.bind("<Leave>", on_leave)

step8 = Button(window, text='消去合取词', relief='flat', command=step8, bg='white')
step8.place(x=2, y=420)
step8.bind("<Enter>", on_enter)
step8.bind("<Leave>", on_leave)

step9 = Button(window, text='子句变元标准化', relief='flat', command=step9, bg='white')
step9.place(x=2, y=485)
step9.bind("<Enter>", on_enter)
step9.bind("<Leave>", on_leave)

window.mainloop()