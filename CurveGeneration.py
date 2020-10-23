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
def Spherical(xyz):
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew = np.zeros_like(xyz)
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,2] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def Cartesian(rab):
    x = rab[:,0]*np.sin(rab[:,1])*np.cos(rab[:,2])
    y = rab[:,0]*np.sin(rab[:,1])*np.sin(rab[:,2])
    z = rab[:,0]*np.cos(rab[:,1])
    return np.vstack((x,y,z))
def BoundPoints(xyz, Bound_Radio):
    rab = Spherical(xyz) # radio, alpha, beta
    rab[rab[:,0]<Bound_Radio[0],0] = Bound_Radio[0]
    rab[rab[:,0]>Bound_Radio[1],0] = Bound_Radio[1]
    xyz = Cartesian(rab)
    return xyz

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

# Brute force PLOTTING =)
def AnimateFigureBrute(fig, file_name="basic_animation.mp4"):
    # Animate
    global vec_vision
    def animate(i):
        global vec_vision
        print(int(i/len(vec_vision)*100),end="\r")
        for ax in fig.axes: ax.remove() #cleaner
        target_point = vec_vision[i]
        print()
        for idx, [user_eq_x, user_eq_y, user_eq_z, guess] in enumerate(user_eqs):
            print("   =>",i, idx+1, end="\r")
            X,Y,Z, results, error = GeneratePoint(user_eq_x, user_eq_y, user_eq_z, guess, init_point, target_point, extraplot, debug=False)
            ax = CreateFigure([cols, rows], idx)
            ax = UpdateFigure(ax,[X,Y,Z], init_point, target_point)
            # ax.view_init(0,90)
            ax.view_init(90,0)
            fig.axes.append(ax)
            targets.append(results)
            errors.append(error)
            # print("###",np.array([X[0],Y[0],Z[0]]))
            # print("==>",np.array([X[-1],Y[-1],Z[-1]]))
            # print()
        return fig,
    # Animate
    from matplotlib import animation
    anim = animation.FuncAnimation(fig, animate, frames=len(vec_vision), interval=1, blit=True)
    anim.save(file_name, fps=30, extra_args=['-vcodec', 'libx264'])


def GeneratePoint(user_eq_x,
                    user_eq_y,
                    user_eq_z,
                    guess,
                    init_point,
                    target_point,
                    extraplot,
                    debug=True):
    global pX,pY,pZ, result
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
        opts = {"maxiter":1000, 'disp': False}
        # bnds = ((-5, 5), (-5, 5), (-5, 5), (None, None)) # bounds=bnds,
        prev_result = np.inf

        for _ in range(3):
            for _ in range(10):
                result = optimize.minimize(Gcurve, guess, tol=1e-5, options=opts)
                prev_result = result.fun
                if prev_result<1e-6: break;
                else: 
                    guess[:-1] = guess[:-1]-0.5
                    # print("new guess:", guess)
            guess[-1] = guess[-1]-1
        targets = result.x
        if debug:
            # print(result)
            print(result.success, "dist:", result.fun)
            print("new guess:", guess)
            print("algorithm output:", targets)

        p_x, p_y, p_z, t = targets
        N = 100;
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
        if debug:
            print("time:",t2-t1)
        dist = result.fun
        pX = np.round(pX,6)
        pY = np.round(pY,6)
        pZ = np.round(pZ,6)
    else:
        if debug:
            print("NO RIGHT FUNCTION", dist)
        pX,pY,pZ = None,None,None
        targets = None
        dist = None
    return pX,pY,pZ, targets, dist

user_eqs = [

[
"p_x*(t+np.sin(t*np.sin(t)))",
"p_y*np.sin(p_z*t)",
"p_z*t",
[1.,1.,1.,   5.],
],


[
"p_x*np.sin(p_x*t)",
"-p_y*(np.cos(p_y*t)-1)",
"p_z*t",
[1,1,1,   5],
],


[
"-p_x* np.sin(p_y*t)",
"p_y*(np.cos(p_x*t)-1)",
"p_z*t",
[1,1,1,   5],
],


[
"p_x* np.sin(t)-0.1*p_z*np.cos(10*t)",
"p_y*(np.cos(0.1*t)-1)",
"p_z*t",
[1,1,1,   5],
],


[
"p_x*(t+np.sin(t*np.sin(4*t)))",
"-p_y*np.sin(p_z*t)",
"p_z*t",
[1,1,1,   5],
],


[
"p_x*(np.sin(2*t)+3*np.sin(p_x*t))",
"p_y*np.sin(p_z*t)",
"t*np.sin(p_z)",
[1,1,1,   0],
],



[
"p_x*np.sin(5*t)-np.sin(t)",
"p_y*np.sin(t)+p_x*t",
"p_z*t",
[1,1,1,   5],
],


[
"p_x*np.sin(t)-0.1*(np.cos(10*t)-1)",
"p_y*np.sin(p_y*t)",
"t*np.sin(p_z)",
[1,1,1,   5],
],


[
"p_x* np.sin(p_x*t)    + p_z*(np.cos(0.1*t)-1)",
"p_y*(np.cos(p_y*t)-1) + p_z* np.sin(0.1*t)",
"p_z*t",
[1,1,1,   5],
],


]

lims = [-3.,3.]
init_point = [1., 1., 1.]
target_point = [-1.1, -1.7, 1.3]
extraplot = 0 # continue plotting the line after reaching the target point
Bound_Radio = [0.5,3.];
# Very standard : 2,4,5,6,7,9

N_cols = 3;
rows, cols = N_cols, (len(user_eqs)-1)//N_cols+1
targets = []; errors = []
fig = CreateTemplate(scale=14, relative_size=[rows, cols])
for idx, [user_eq_x, user_eq_y, user_eq_z, guess] in enumerate(user_eqs):
    print(idx+1)
    X,Y,Z, results, error = GeneratePoint(user_eq_x, user_eq_y, user_eq_z, guess, init_point, target_point, extraplot)
    
    X,Y,Z = BoundPoints(np.array([X,Y,Z]).T, Bound_Radio)

    ax = CreateFigure([cols, rows], idx)
    ax = UpdateFigure(ax,[X,Y,Z], init_point, target_point)
    # ax.view_init(0,90)
    ax.view_init(90,0)
    fig.axes.append(ax)
    targets.append(results)
    errors.append(error)
    print("###",np.array([X[0],Y[0],Z[0]]))
    print("==>",np.array([X[-1],Y[-1],Z[-1]]))
    print()
plt.show()

for idx in range(len(targets)): print(idx+1, targets[idx], errors[idx])
# AnimateFigure(fig)



"""
# Not recommended!!! or change the steps to 0.5 for faster results. Otherwise it may take at least 8 hours
vec_vision = []
initY = -3.; endY = 3.; stepY = 0.1
initZ = 0.5; endZ =2.5; stepZ = 0.1
for xx in np.arange(-3.,3.,0.2):
    for yy in np.arange(initY, endY, stepY):
        for zz in np.arange(initZ, endZ, stepZ):
            vec_vision.append([xx,yy,zz])
        temp = initZ
        initZ = endZ
        endZ = temp
        stepZ *= -1.
    temp = initY
    initY = endY
    endY = temp
    stepY *= -1.

# Very standard : 2,4,5,6,7,9
N_cols = 3
rows, cols = N_cols, (len(user_eqs)-1)//N_cols+1
targets = []; errors = []
fig = CreateTemplate(scale=14, relative_size=[rows, cols])
AnimateFigureBrute(fig)
"""
