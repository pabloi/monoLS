%% basic tests:
p=2;
%% fit line
a=randn;
y=randn(1,500)+a*[1:500];
z=monoLS(y,p,0,0);

figure
subplot(1,2,1)
hold on
plot(y,'x')
plot(z,'LineWidth',2)


%% Noisy negative exponential
y1=randn+randn*exp(-[0:500]/abs(100*randn));
y1=y1(:);
y=y1+.1*randn(size(y1));

reg=5;
tic
z=monoLS(y,p,0,0);
toc
tic
z2=monoLS(y,p,1,0);
toc
tic
z3=monoLS(y,p,1,reg);
toc
tic
z4=monoLS(y,p,2,reg);
toc
%tic
%z5=monoLS(y,p,3,reg); %Fails to converge
%toc
subplot(1,2,2)
hold on
plot(y,'x')
plot(z,'LineWidth',2,'DisplayName',['Monotonic (f'' \geq 0), e=' num2str(norm(z-y))])
plot(z2,'LineWidth',2,'DisplayName',['Double-Monotonic (f'''' \geq 0), e=' num2str(norm(z2-y))])
plot(z3,'LineWidth',2,'DisplayName',['Regularized Double-Monotonic (f'''' \geq 0), e=' num2str(norm(z3-y))])
plot(z4,'LineWidth',2,'DisplayName',['Regularized Triple-Monotonic (f'''''' \geq 0), e=' num2str(norm(z4-y))])
%plot(z5,'LineWidth',2,'DisplayName',['Regularized Quadruple-Monotonic (f'''''' \geq 0), e=' num2str(norm(z5-y))])
plot(y1,'LineWidth',2,'DisplayName',['Generator, e=' num2str(norm(y1-y))])
legend

%% Compare monoLS and monoLS2
