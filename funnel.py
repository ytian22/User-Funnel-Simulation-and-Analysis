
# coding: utf-8

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def UserSim(n, lambd):
    '''
    This function simulates the survival time of n users. 
    Here the survival time is the expected duration of time until a user quits the process.

    inputs:
    n: the number to of users to simulate
    lambd: the parameter

    output: a list of exponential random variable simulations
    '''
    user_quit_time = np.random.exponential(1.0/lambd, n)
    return user_quit_time.tolist()

def survive_number(user_quit_time, threshold):
    '''
    This function calculates the number of users who survive beyond the time hurdle.
    '''
    return  len([i for i in user_quit_time if i > threshold])

def autolabel(rects):
    '''
    This function attaches a text label above each bar displaying its height.
    '''
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')

def EstLam1(user_quit_time):
    '''
    This function uses the output of UserSim to estimate the that generated those values.
    '''
    return 1.0/np.mean(user_quit_time)

def HurdleFun(user_quit_time, breakpoints):
    '''
    This function takes a list of breakpoints and user quit times and returns the number of users who made it to that breakpoint.

    inputs:
    user_quit_time: a list from UserSim of times that users quit
    breakpoints: a list of breakpoints (of arbitrary length)

    output: a list of exponential random variable simulations
    '''
    user_quit_time = np.array(user_quit_time)
    my_hist = np.ones(len(breakpoints)+1)
    previous = 0
    
    for i in range(len(breakpoints)):
        my_hist[i] = (user_quit_time<=breakpoints[i]).sum() - previous
        previous += my_hist[i]
    
    my_hist[-1] = (user_quit_time>breakpoints[i]).sum()
    
    return my_hist

def EstLam2(sim_list, breaks):
    '''
    This function returns a python lambda function PRT which takes, as its sole input, a value 
    and which will return the log likelihood value for the data originally entered in HurdleFun.

    inputs:
    sim_list: a simulated list (the output of HurdleFun)
    breaks: a list of breakpoints, which is the same input of HurdleFun
    '''
    def PRT(lambd):
        sum1 = sim_list[0]*np.log(1-np.exp(-lambd*breaks[0]))
        
        sum2 = 0
        for i in range(1, len(sim_list)-1):
            sum2 += sim_list[i]*np.log(np.exp(-lambd*breaks[i-1]) - np.exp(-lambd*breaks[i]))
            
        sum3 = -sim_list[-1]*lambd*breaks[-1]
        
        return sum1+sum2+sum3
    return PRT

def MaxMLE(sim_list, breaks, lambd_list):
    '''
    This function takes in a range of lambda values and then searches over that space for the maximum likelihood. 
    This function returns the lambda value that maximizes the likelihood.
    '''
    PRT = EstLam2(sim_list, breaks)
    idx = np.argmax([PRT(l) for l in lambd_list])
    return lambd_list[idx]


if __name__ == '__main__':

    # 1.(a)
    # Create a funnel visualization with 1,000 users and a parameter of 2 with stops every .25 to 3.
    user_quit_time = UserSim(1000, 2)
    time_list = np.arange(0, 3.25, 0.25)
    funnel = [survive_number(user_quit_time, t) for t in time_list]
    
    # visualization
    fig, ax = plt.subplots(figsize=(7,5))
    rects = ax.bar(range(len(funnel)), funnel)
    ax.set_xticks(range(0, len(funnel),2))
    ax.set_xticklabels(np.arange(0, 3.25, 0.5));
    ax.set_xlabel('time elapsed')
    ax.set_ylabel('users survived')
    ax.set_title('Funnel with lambda=2')
    autolabel(rects)
    plt.show()


    # 1.(b) 
    # Repeat the previous assignment, but with lambda equal to the values of .2 to 3.0 instep sizes of .2 and plot each.
    plt.figure(figsize=(8,6))
    time_list = np.arange(0, 3.1, 0.1)

    for lambd in np.arange(0.2, 3.2, 0.2):
        user_quit_time = UserSim(1000, lambd)
        funnel = [survive_number(user_quit_time, t) for t in time_list]
        plt.plot(time_list, funnel, label='$\lambda=%.2f$' % lambd)
    
    plt.legend(bbox_to_anchor=(1,1))
    plt.xlabel('time elapsed')
    plt.ylabel('user survived')
    plt.title('Funnel with different lambda')
    plt.show()


    # 2.(b) 
    # Generate a sample of 1,000 users using UserSim with lambda equal to 1 and estimate lambda using EstLam1.
    user_quit_time = UserSim(1000, 1)
    print(EstLam1(user_quit_time))


    # 2.(c)
    # Using that same sample of 1,000 users, bootstrap a 95% confidence interval for the lambda estimate (use 500 bootstraps).
    lambd_hat_list = []
    for _ in range(500):
        lambd_hat_list.append(EstLam1(np.random.choice(user_quit_time, size=1000, replace=True)))
    lower, upper = np.percentile(lambd_hat_list, [2.5, 97.5])
    print('CI Lower Bound: %s, CI Upper Bound: %s'%(lower, upper))       


    # 2.(d) 
    # Repeat the above process with number of users equal to 100, 200, 500, 1,000, 2,000, 5,000 and 10,000.
    estimate_list = []
    lower_list = []
    upper_list = []

    n_users = [100, 200, 500, 1000, 2000, 5000, 10000]
    for i in n_users:
        user_quit_time = UserSim(i, 1)
        
        lambd_hat_list = []
        for _ in range(500):
            lambd_hat_list.append(EstLam1(np.random.choice(user_quit_time, size=i, replace=True)))
        
        estimate_list.append(np.mean(lambd_hat_list))
        lower_list.append(np.percentile(lambd_hat_list, 2.5))
        upper_list.append(np.percentile(lambd_hat_list, 97.5))
    
    # Create a table which includes both the estimated and the confidence intervals.
    table1 = pd.DataFrame({'n_users': n_users, 
                           'estimation': estimate_list,
                           'lower': lower_list,
                           'upper': upper_list})[['n_users', 'estimation', 'lower', 'upper']]
    table1['interval'] = table1['upper'] - table1['lower']
    print(table1)

    # Create a visualization which includes both the estimated and the confidence intervals.
    plt.fill_between(table1['n_users'], table1['lower'], table1['upper'])
    plt.plot(table1['n_users'], table1['estimation'], 
             label='$estimated \; \lambda$',
             c='orange')
    plt.xscale('log')
    plt.title('confidence interval vs n_user')
    plt.xlabel('n_user(log scale)')
    plt.ylabel('confidence interval')
    plt.legend()
    plt.show()

    # 4.(a)
    # Using the functions defined above, run 1,000 simulations of 100 users 
    # with the following sets of break points [0.25, 0.75], [0.25, 3], [0.25, 10].
    output = []
    for lambd in np.arange(0.5, 4.5, 0.5):
        avg_diff_list = []
        for breaks in [[0.25, 0.75], [0.25, 3], [0.25, 10]]: 
            diff_list = []
            for _ in range(1000):
                user_quit_time = UserSim(100, lambd)
                lambd1 = EstLam1(user_quit_time)
                lambd2 = MaxMLE(HurdleFun(user_quit_time, breaks), 
                                breaks, 
                                np.arange(0.1, 3, 0.05))
                diff = np.abs(lambd1 - lambd2)
                diff_list.append(diff)
            avg_diff_list.append(np.mean(diff_list))
        output.append(avg_diff_list)

    columns=['2nd_brk_0.75', '2nd_brk_3', '2nd_brk_10']
    df = pd.DataFrame(np.array(output), columns=columns)
    df['lambda'] = np.arange(0.5, 4.5, 0.5)
    table2 = df[['lambda'] + columns]
    print(table2)

    # 4.(b)
    # Run additional simulations to identify other trends.
    output = []
    for lambd in np.arange(0.5, 4.5, 0.5):
        avg_diff_list = []
        for breaks in [[0.25, 0.50], [0.25, 0.75], [0.25, 1],
                       [0.25, 2], [0.25, 3], [0.25, 5], [0.25, 10]]: 
            diff_list = []
            for _ in range(1000):
                user_quit_time = UserSim(100, lambd)
                lambd1 = EstLam1(user_quit_time)
                lambd2 = MaxMLE(HurdleFun(user_quit_time, breaks), 
                                breaks, 
                                np.arange(0.1, 3, 0.05))
                diff = np.abs(lambd1 - lambd2)
                diff_list.append(diff)
            avg_diff_list.append(np.mean(diff_list))
        output.append(avg_diff_list)

    columns=['2nd_brk_0.5', '2nd_brk_0.75', '2nd_brk_1', '2nd_brk_2', '2nd_brk_3', '2nd_brk_5', '2nd_brk_10']
    df = pd.DataFrame(np.array(output), columns=columns)
    df['lambda'] = np.arange(0.5, 4.5, 0.5)
    table3 = df[['lambda'] + columns]
    print(table3)
    
    # visualization
    plt.figure(figsize=(9,5))
    x = range(len(columns))
    for lbd, row in zip(table3['lambda'], table3.values[:, 1:]):
        plt.plot(x, row, label='$\lambda=%s$'%lbd)
    plt.xticks(x, columns)
    plt.xlabel('break points')
    plt.ylabel('average differene')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()