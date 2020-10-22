import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy import optimize
import time
np.set_printoptions(suppress=True)
'''
References:
https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
https://stackoverflow.com/questions/13670333/multiple-variables-in-scipys-optimize-minimize
http://cudaopencl.blogspot.com/2013/02/3d-parametric-curves-are-cool.html
https://christopherchudzicki.github.io/MathBox-Demos/parametric_curves_3D.html

'''

def CreateTemplate(scale, relative_size):
    fig = plt.figure(figsize=(scale, scale*relative_size[1]/relative_size[0]))
    fig.tight_layout()
    plt.subplots_adjust(top = 0.93, bottom = 0.0, right = 1, left = 0, hspace = 0.1, wspace = 0.1)
    fig.suptitle("Camera trajectory", fontsize=18)
    return fig
def CreateFigure(grid, idx):
    # ax = fig.gca(projection='3d')
    # ax = fig.add_axes([0,0,1,1], projection='3d')
    ax = plt.subplot2grid((grid[0], grid[1]), (idx//N_cols, idx%N_cols), projection='3d')
    ax.title.set_text("Curve {}".format(idx+1))
    return ax
def UpdateFigure(ax, curve, init_point, target_point):
    X,Y,Z = curve
    ax.plot([0],[0],[0], marker="o", color="k", markersize=10, label='target objects')
    ax.plot([init_point[0]],[init_point[1]],[init_point[2]], marker="o", color="b",markersize=7, label='initial point')
    ax.plot([target_point[0]],[target_point[1]],[target_point[2]], marker="o", color="r", markersize=7, label='target point')
    ax.plot(X,Y,Z, label='parametric curve')
    ax.set_xlabel('X', fontsize=18)
    ax.set_ylabel('Y', fontsize=18)
    ax.set_zlabel('Z', fontsize=18)
    ax.legend(loc = "upper right")
    ax.set_xticks(np.arange(lims[0],lims[1]))
    ax.set_yticks(np.arange(lims[0],lims[1]))
    ax.set_zticks(np.arange(lims[0],lims[1]))
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_zlim(lims)
    return ax

def AnimateFigure(fig, file_name="basic_animation.mp4"):
    # Animate
    global vec_vision
    def animate(i):
        global vec_vision
        print(int(i/len(vec_vision)*100),end="\r")
        for ax in fig.axes:
            ax.view_init(vec_vision[i][0], vec_vision[i][1])
        return fig,

    vec_vision = []; step = 1
    # rotate the axes and update
    start_X = 0; end_X = 80; 
    start_Y = 0; end_Y = 80; 
    for angle in np.arange(start_X, end_X, step): # horizontal
        vec_vision.append([start_Y, angle])
    for angle in np.arange(start_Y, end_Y,step): # vertical
        vec_vision.append([angle, end_X])
    for angle in np.arange(end_X, start_X, -step): # horizontal
        vec_vision.append([end_Y, angle])
    for angle in np.arange(end_Y, start_Y,-step): # vertical
        vec_vision.append([angle, start_X])
    # Animate
    from matplotlib import animation
    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(vec_vision), interval=1, blit=True)
    anim.save(file_name, fps=30, extra_args=['-vcodec', 'libx264'])

def GeneratePoint(user_eq_x,
                    user_eq_y,
                    user_eq_z,
                    guess,
                    init_point,
                    target_point,
                    extraplot):
    global pX,pY,pZ
    exec("""
global gX; 
def gX(t=0, p_x=0, p_y=0, p_z=0):
    return {0}""".format(user_eq_x))
    exec("""
global gY; 
def gY(t=0, p_x=0, p_y=0, p_z=0):
    return {0}""".format(user_eq_y))
    exec("""
global gZ; 
def gZ(t=0, p_x=0, p_y=0, p_z=0):
    return {0}""".format(user_eq_z))
    exec("""
global Gcurve
def Gcurve(params):
    p_x, p_y, p_z, t = params
    [pointx, pointy, pointz] = [init_point[0]+{0}, init_point[1]+{1}, init_point[2]+{2}]
    dist = np.array([pointx,pointy,pointz])-{3}
    # return np.linalg.norm(dist) # it generates a false result =S
    return (dist**2).sum()
    """.format(user_eq_x, user_eq_y, user_eq_z, target_point))

    dist = gX()+gY()+gZ()
    if dist==0 and not np.isnan(dist):
        t1 = time.time()
        guess = np.float32(guess)
        opts = {"maxiter":1000, 'disp': True}
        # bnds = ((-5, 5), (-5, 5), (-5, 5), (None, None)) # bounds=bnds,
        prev_result = np.inf

        for i in range(10):
            result = optimize.minimize(Gcurve, guess, tol=1e-5, options=opts)
            prev_result = result.fun
            if prev_result<1e-6: break;
            else: 
                guess[:-1] = guess[:-1]-0.5
                # print("new guess:", guess)
        targets = result.x
        # print(result)
        print()
        print(result.success, "dist:", dist)
        print("new guess:", guess)
        print("algorithm output:", targets)

        p_x, p_y, p_z, t = targets
        N = 200;
        plot_point = []
        for i,(axis,eq) in enumerate(zip(["X","Y","Z"],[user_eq_x, user_eq_y, user_eq_z])):
            args = "global p{0}; p{0} = {1}+g{0}(np.linspace(0,targets[3]+{2},N)".format(axis,init_point[i],extraplot)
            for i, param in enumerate(["p_x","p_y","p_z"]):
                if param in eq: args += ","+str(targets[i])
                else: args += ",0"
            args += ")"
            plot_point.append(args)  
        exec(plot_point[0])
        exec(plot_point[1])
        exec(plot_point[2])
        t2 = time.time()
        print("time:",t2-t1)
    else:
        # print(dist)
        pX,pY,pZ = None,None,None
    return pX,pY,pZ, targets

lims = [-3,3]
init_point = [1,1,1]
target_point = [-2, -2, 2]
extraplot = 0 # continue plotting the line after reaching the target point
# function must start in 0,0,0
user_eqs = [
[
"p_x*np.sin(t)",
"p_y*np.sin(t)",
"p_z*t",
[0,0,0,   3],
],

[
"p_x*np.sin(p_x*t)",
"-p_y*(np.cos(p_y*t)-1)",
"p_z*t",
[1,1,1,   2],
],

[
"p_x*np.sin(p_x*t)",
"-p_y*(np.cos(p_y*t)-1)",
"p_z*t",
[5,5,5,   2],
],

[
"t* np.sin(p_x*t)",
"t*(np.cos(p_y*t)-1)",
"p_z*t",
[5,5,5,   2],
],

[
"p_x* np.sin(p_z*t)",
"p_y*(np.cos(p_z*t)-1)",
"p_z**t-1",
[1,1,1,   5],
],

[
"p_x* np.sin(p_y*t)",
"p_y*(np.cos(p_x*t)-1)",
"p_z*t",
[0,-1,1,   5],
],

[
"p_x* np.sin(p_y*t)",
"p_y*(np.cos(p_x*t)-1)",
"p_z*t",
[0,1,1,   5],
],

[
"p_z* np.sin(p_x*t)    + p_z*(np.cos(p_y*t)-1)",
"p_z*(np.cos(p_y*t)-1) + p_z* np.sin(p_x*t)",
"p_z*t",
[-1,3,1,   5],
],


[
"p_z* np.sin(p_x*t)    + p_z*(np.cos(p_y*t)-1)",
"p_z*(np.cos(p_y*t)-1) + p_z* np.sin(p_x*t)",
"p_x*p_y*t",
[-3,2,1,   3],
],


]
N_cols = 3
rows, cols = N_cols, (len(user_eqs)-1)//N_cols+1
targets = []
fig = CreateTemplate(scale=14, relative_size=[rows, cols])
for idx, [user_eq_x, user_eq_y, user_eq_z, guess] in enumerate(user_eqs):
    X,Y,Z, results = GeneratePoint(user_eq_x, user_eq_y, user_eq_z, guess, init_point, target_point, extraplot)
    targets.append(results)
    ax = CreateFigure([cols, rows], idx)
    ax = UpdateFigure(ax,[X,Y,Z], init_point, target_point)
    fig.axes.append(ax)
plt.show()

AnimateFigure(fig)