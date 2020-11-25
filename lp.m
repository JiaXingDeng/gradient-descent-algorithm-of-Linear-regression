function[wt,t,s]=lp(x,y,w0,a)
%%
%使用随机梯度下降法SGD进行线性回归，其中：
% x指自变量,y指目标变量，a指梯度下降学习率，w0为初始值（不能为[0,0]'）
% 输出中wt指最终得出的直线参数，t指迭代次数，s指当前loss函数值
% EXAMPLE：
% load('lp.mat');
%[wt,t]=lp(x,y,[1,1]',0.01)
l=length(y);
x1=[x,ones(l,1)];%数据加列
WX=x1*w0;
s=1/2*(WX-y)'*(WX-y);%计算第一次损失函数
t=0;
loss=[];%初始值设定
while s>0.001 && t<200 %终止条件
        i=round(rand(1,1)*(l-1)+1);%随机选择一组数
        wt=w0+a*w0*(y(i)-WX(i))*x(i)';%迭代
        w0=wt;
        WX=x1*w0;
        s=1/2*(y-WX)'*(y-WX);%计算损失函数
        t=t+1;
        loss(t) = s;
end
%%
%绘图
subplot(211)
plot(x,y,'rx','linewidth',1.5);
hold on;
grid on;
p1 = 0;
p2 = wt(2);
q1 = 8;
q2 = wt(1)*q1+wt(2);
plot([p1 q1],[p2 q2],'k-','linewidth',1.8)
axis([0 8 0 3])
set(gca,'position',[0.04 0.55 0.94 0.43]) 
xlabel('Petal.Length')
ylabel('Petal.Width')
hold off;
subplot(212)
tt = 1:10:t;
plot(tt,loss(tt),'b-','linewidth',1.5);
xlabel('epoch')
ylabel('loss')
set(gca,'position',[0.04 0.07 0.94 0.42]) 
grid on;

    
