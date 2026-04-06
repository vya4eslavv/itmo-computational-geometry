import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
np.set_printoptions(precision=3, suppress=True)
def apply_transform(points, M):
    pts = np.hstack([points, np.ones((len(points), 1))])
    res = pts @ M.T
    return res[:, :2] / res[:, 2:3]
def plot_polygon(ax, pts, title="", labels=None):
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.add_patch(Polygon(pts, closed=True, fill=False))
    ax.scatter(pts[:, 0], pts[:, 1])
    if labels is not None:
        for (x, y), lab in zip(pts, labels):
            ax.text(x, y, lab)
def T(a):
    ax, ay = a
    return np.array([[1, 0, ax],
                     [0, 1, ay],
                     [0, 0, 1]], float)

def R_about(p, phi):
    x, y = p
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, x - c*x + s*y],
                     [s,  c, y - s*x - c*y],
                     [0,  0, 1]], float)

def S_Ox():
    return np.array([[1, 0, 0],
                     [0, -1, 0],
                     [0, 0, 1]], float)

def S_Oy():
    return np.array([[-1, 0, 0],
                     [0,  1, 0],
                     [0,  0, 1]], float)

def reflect_line_through_origin(theta):
    c = np.cos(2*theta)
    s = np.sin(2*theta)
    return np.array([[ c, s, 0],
                     [ s,-c, 0],
                     [ 0, 0, 1]], float)

def H_about(p, k):
    x, y = p
    return np.array([[k, 0, (1-k)*x],
                     [0, k, (1-k)*y],
                     [0, 0, 1]], float)

P1 = np.array([0., 0.])
P2 = np.array([6., 1.])
P3 = np.array([2., 5.])
tri = np.vstack([P1, P2, P3]) #треуг как массив 3 на 2
tri_hom = np.hstack([tri, np.ones((3, 1))]) #делаю однород коорд

a_vec = (2, 1)            # перенос
phi = np.deg2rad(30)      # поворот в радинанах задаю
k = 0.8                   # гомотетия относительно O
m = 1.2                   # гомотетия относительно M
Sl = reflect_line_through_origin(np.pi/4)

# центроид
C = tri.mean(axis=0) #№среднее по столбцам => ( (x1+x2+x3)/3 , (y1+y2+y3)/3 )

pairs = [(0,1), (1,2), (2,0)]
lens = [(np.linalg.norm(tri[i]-tri[j]), (i,j)) for i,j in pairs]
(i_min, j_min) = min(lens, key=lambda t: t[0])[1]
M_mid = (tri[i_min] + tri[j_min]) / 2

Ta  = T(a_vec)
RC  = R_about(C, phi)
SOx = S_Ox()
SOy = S_Oy()
HOk = H_about((0,0), k)
HMm = H_about(M_mid, m)

F_tri = HMm @ HOk @ Sl @ SOy @ SOx @ RC @ Ta

steps_names = ["Исходный", "Ta", "RC", "SOx", "SOy", "Sl", "HOk", "HMm (итог)"]
mats = [np.eye(3), Ta, RC, SOx, SOy, Sl, HOk, HMm]

tri_steps = [tri]
acc = np.eye(3)
for M in mats[1:]:
    acc = M @ acc
    tri_steps.append(apply_transform(tri, acc))

print("лаба 1 задание 1")
print("Треугольник (декартовы):\n", tri)
print("Треугольник (однородные координаты):\n", tri_hom)
print("\nМатрицы преобразований:")
print("Ta=\n", Ta)
print("RC=\n", RC)
print("SOx=\n", SOx)
print("SOy=\n", SOy)
print("Sl (y=x)=\n", Sl)
print("HOk=\n", HOk)
print("HMm=\n", HMm)
print("\nИтоговая матрица композиции F_tri=\n", F_tri)
print("\nИтоговый образ треугольника:\n", tri_steps[-1])

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs = axs.flatten()

all_pts = np.vstack(tri_steps)
xmin, ymin = all_pts.min(axis=0) - 1
xmax, ymax = all_pts.max(axis=0) + 1

for i, ax in enumerate(axs):
    plot_polygon(ax, tri_steps[i], title=f"{steps_names[i]}", labels=["P1","P2","P3"])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

fig.tight_layout()
fig.savefig("picture_1.png", dpi=200)
plt.show()

#2 задание
K = np.array([-1., -1.])
L = np.array([ 1., -1.])
M = np.array([ 1.,  1.])
N = np.array([-1.,  1.])
square = np.vstack([K, L, M, N])
square_hom = np.hstack([square, np.ones((4,1))])

#  параллелограмм ABCD
# A на луче KM и в 3 раза дальше от K, чем точка M => A = K + 3(M-K)
A = K + 3*(M - K)              # (5,5) (в 3 раза дальше тройка кэф t)
# AD параллелен KN и в 2 раза длиннее KN
D = A + 2*(N - K)
# N-K — вектор KN
# 2*(N-K) — вектор в 2 раза длиннее и параллельный

angle_AB = np.pi/6
# задаю угол направления AB к оси Ox

dir_AB = np.array([np.cos(angle_AB), np.sin(angle_AB)])
# единичный вектор направления AB

length_AB = 4 / np.sin(np.pi/3)
# вычисляем длину AB (высота 4 при п/3)

AB = length_AB * dir_AB #вектор аb

B = A + AB

C = B + (D - A)
# BC параллелен AD

ABCD = np.vstack([A, B, C, D])

# F(x,y) = (a x + b y + e,  c x + d y + f)

A_sys = np.array([[-1,-1,1],
                  [ 1,-1,1],
                  [ 1, 1,1]], float)




bx = np.array([A[0], B[0], C[0]], float)

by = np.array([A[1], B[1], C[1]], float)

a, b, e = np.linalg.solve(A_sys, bx)
#решение систем

c, d, f = np.linalg.solve(A_sys, by)

F = np.array([[a, b, e],
              [c, d, f],
              [0, 0, 1]], float)

F_inv = np.linalg.inv(F)
# находим обратную матрицу

img_square = apply_transform(square, F)
# образ

back_square = apply_transform(ABCD, F_inv)
#  обратное к параллелограмму

print("\n лаба 1 задание 2")
print("Квадрат (однородные координаты):\n", square_hom)
print("\nМатрица F=\n", F)
print("\nМатрица F^{-1}=\n", F_inv)
print("\nF(KLMN)=\n", img_square)
print("\nF^{-1}(ABCD)=\n", back_square)

err = np.linalg.norm(img_square - ABCD)

print("\nПроверка (норма ошибки F(KLMN)-ABCD):", err)
print("Проверка (округлённо):", round(err, 12))

fig, ax = plt.subplots(figsize=(7,7))

ax.set_aspect("equal")                 # равный масштаб осей
ax.grid(True, alpha=0.3)               # сетка

ax.add_patch(Polygon(square, closed=True, fill=False))

ax.scatter(square[:,0], square[:,1])

for p, lbl in zip(square, ["K","L","M","N"]):
    ax.text(p[0], p[1], lbl)

ax.add_patch(Polygon(img_square, closed=True, fill=False))

ax.scatter(img_square[:,0], img_square[:,1])

for p, lbl in zip(img_square, ["A","B","C","D"]):
    ax.text(p[0], p[1], lbl)

ax.set_title("Квадрат KLMN и образ F(KLMN)=ABCD")
fig.tight_layout()
fig.savefig("picture_2.png", dpi=200)
plt.show()

A = F[:2, :2]
t = F[:2,  2]

A_inv = np.linalg.inv(A)
t_inv = -A_inv @ t

F_inv_manual = np.eye(3)
F_inv_manual[:2, :2] = A_inv
F_inv_manual[:2,  2] = t_inv

print("F_inv\n", F_inv_manual)
print("F_inv (\n", F_inv)


fig, ax = plt.subplots(figsize=(6,6))

ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

ax.add_patch(Polygon(tri_steps[-1], closed=True, fill=False))
ax.scatter(tri_steps[-1][:,0], tri_steps[-1][:,1])

for p, lbl in zip(tri_steps[-1], ["P1'","P2'","P3'"]):
    ax.text(p[0], p[1], lbl)

ax.add_patch(Polygon(tri, closed=True, fill=False))
ax.scatter(tri[:,0], tri[:,1])

for p, lbl in zip(tri, ["P1","P2","P3"]):
    ax.text(p[0], p[1], lbl)

ax.set_title("исходный и обратный")

plt.show()