function[wt,t,s]=RMSProp(x,y,w0,u,a,e)
%%
%ʹ��RMSProp���������Իع飬���У�
% xָ�Ա���, yָĿ�������w0Ϊ��ʼֵ������Ϊ[0,0]'��,aָѧϰ��,uָ��������
% e��Ϊά�������ȶ��Զ���ӵĳ���
% EXAMPLE��
% load('lp.mat');
%[wt,t,s]=RMSProp(x,y,[0,0]',0.9,0.01,10^(-6))
l=length(y);
x1=[x,ones(l,1)];%���ݼ���
WX=x1*w0;
s=1/2*(WX-y)'*(WX-y);%�����һ����ʧ����
t=0;
v=zeros(2,1);%������ֵΪ0
loss=[];
while s>0.00001 && t<200
        gammax=2/l*(-x1'*y+x1'*WX);%�ݶȼ���
        vt=u*v+(1-u)*gammax.*gammax;%��������
        v=vt;
        wt=w0-a*(gammax./(v.^(1/2)+e));%��������
        w0=wt;
        WX=x1*w0;
        s=1/2*(y-WX)'*(y-WX);
        t=t+1;
        loss(t)=s;
end
%%
%��ͼ
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

    