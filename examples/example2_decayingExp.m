%%
x=1:50;
a=2*rand;
b=randn;
c=length(x)*rand/2;
a2=2*rand;
c2=length(x)*rand/2;
yy=a*exp(-x/c)+b + a2*exp(-x/c2);
y=yy+.1*randn(size(yy)); %Gaussian noise
out=randi(length(y),5,1);
y(out)=randn(5,1); %Some outlier samples

figure;
for normP=1:2; %first and second order (least-squares) fitting
derN=0; %Only enforcing monotonicity
regN=2; %regularizing 2 samples
[z] = monoLS(y,normP,derN,regN);


derN=1; %Enforcing monotonicity with decreasing (absolute) slopes
regN=2; %regularizing 2 samples
oddSign=[];
evenSign=-1;
[z1] = monoLS(y,normP,derN,regN,oddSign,evenSign);

derN=2; %Enforcing monotonicity with decreasing (absolute) slopes, and decreasing (absolute) slope variation
regN=2; %regularizing 2 samples
oddSign=[];
evenSign=-1;
[z2] = monoLS(y,normP,derN,regN,oddSign,evenSign);
[z2b] = monoLS(z,normP,derN,regN,oddSign,evenSign); %enforcing from the monotonic fit


subplot(1,2,normP)
hold on;
plot(x,y,'.') %data
plot(x,yy,'DisplayName','True line')
plot(x,z,'DisplayName','Monotonic LS fit')
plot(x,z1,'DisplayName','Enforcing second derivative sign')
plot(x,z2,'DisplayName','Enforcing third derivative sign')
plot(x(out),y(out),'ro','DisplayName','Outlier datapoints')
%plot(x,z2b,'DisplayName','1+3')
legend
title(['Best monotonic fits using ' num2str(normP) '-norm'])
%axis([min(x) max(x) min(yy) max(yy)])

end