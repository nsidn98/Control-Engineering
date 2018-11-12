w = 0.8
s = tf('s')
G = 19*exp(-2.56*s)/(1+0.25*s)
abc = angle(evalfr(G,w*j))*180/pi+180
beta = abs(evalfr(G,w*j))
T = 50/w
Gc=(1+T*s)/(1+beta*T*s)
Gt = G*Gc
[a,b,c,d] = margin(Gt)
r=feedback(Gt,1)
figure()
step(r)
figure()
bode(G,Gc,Gt)
legend("G",'Gc','Gt')