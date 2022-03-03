def Qlearn(Q,s1,action,rew,disct,s2,alpha):

    target = rew + disct * Q[s2[0],s2[1],0,:].max()

    delta = Q[s1[0],s1[1],0,action] - target
    qdrop = Q[s1[0],s1[1],0,action] - disct*Q[s2[0],s2[1],0,:].max()
    #ratio = Q[s1[0],s1[1],0,action] / (disct*Q[s2[0],s2[1],0,:].max() + 0.00001)
    ratio = Q[s1[0],s1[1],0,:].max() / (disct*Q[s2[0],s2[1],0,:].max() + 0.00001)
    if rew == 1:
        ratio = 0

    # Q learn update
    if Q[s1[0],s1[1],0,action] - alpha*delta > 1:
        c=1
        pass
    Q[s1[0],s1[1],0,action] = Q[s1[0],s1[1],0,action] - alpha*delta

    return Q, delta, qdrop, ratio
