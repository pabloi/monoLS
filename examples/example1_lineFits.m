
%%
x=[-1:.01:1]*rand;
a=randn;
b=randn;
y=a*x+b;
y=y+.1*randn(size(y)); %Gaussian noise
out=randi(length(y),5,1);
y(out)=randn(5,1); %Some outlier samples
normP=2; %least-squares fitting
derN=0; %Only enforcing monotonicity
regN=2; %regularizing 2 samples
[z] = monoLS(y,normP,derN,regN);

derN=1; %Enforcing piece-wise linearity with decreasing (absolute) slopes
regN=2; %regularizing 2 samples
[z1] = monoLS(y,normP,derN,regN);

derN=1; %Enforcing piece-wise linearity with decreasing (absolute) slopes
regN=2; %regularizing 2 samples
oddSign=[];
evenSign=-1;
[z2] = monoLS(y,normP,derN,regN,oddSign,evenSign);

derN=1; %Enforcing piece-wise linearity with decreasing (absolute) slopes
regN=2; %regularizing 2 samples
oddSign=[];
evenSign=1;
[z3] = monoLS(y,normP,derN,regN,oddSign,evenSign);

pp=polyfit(x,y,1);
z4=pp(1)*x+pp(2);

figure;
hold on;
plot(x,y,'.') %data
plot(x,a*x+b,'DisplayName','True line')
plot(x,z,'DisplayName','Monotonic LS fit')
plot(x,z1,'DisplayName','Piecewise linear fit')
%plot(x,z2,'DisplayName','Piecewise linear fit enforcing convergence')
%plot(x,z3,'DisplayName','Piecewise linear fit enforcing divergence')
plot(x,z4,'DisplayName','Line LS fit')
plot(x(out),y(out),'ro','DisplayName','Outlier datapoints')
legend