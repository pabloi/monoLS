function [z] = monoLS(y,p,monotonicDerivativeFlag,regularizeFlag,forceSign)
%This function does an p-norm minimization of (z-y), subject to z being monotonic
%(or constant?). The returned vector z is a 'smoothed' version of y.
%INPUTS:
%y: the column vector or matrix to smooth (monoLS acts along dim 1)
%p: norm used for minimization. Default p=2 (least squares)
%monotonicDerivativeFlag: order of the derivatives forced to be of constant sign. =0 means
%monotonic z, =1 means monotonic z & monotonic derivative (e.g. concave or
%convex all the way), =2 forces the second derivative to be of constant
%sign too, etc.
%regularizeFlag: number of samples that are force to have a zero value for
%the last derivative forced by monotonicDerivativeFlag. These samples are
%taken at the end of the z if y is increasing, and at the beginning of z if
%it is decreasing. It avoids overfit of said samples.
%OUTPUT:
%z=Best-fit approximation of data given constraints
%Notice that a monotonic best-fit is always piece-wise constant (derivative is null almost everywhere) in presence
%of noise (i.e. if data is not truly monotonic). In the same way, if more
%derivatives are enforced, the last enforced one will be piece-wise
%constant, so one order less will be piece-wise linear, two orders less
%will be piece-wise quadratic and so forth.

%TODO:
%1) Confim that the optimization (p=2) works & is fast (against some good optimization library?)
%4) Add optimization over both increasing and decreasing functions, and
%choose the best fit (still according to p-norm) over both.
%5) Fix convergence for p~=2 (?)

%% ARGUMENT CHECK:
if nargin<2 || isempty(p)
    p=2;
end
if nargin<3 || isempty(monotonicDerivativeFlag)
    monotonicDerivativeFlag=0;
elseif monotonicDerivativeFlag>numel(y)
    error('Cannot force the sign of so many derivatives!')
end
if nargin<4 || isempty(regularizeFlag) || monotonicDerivativeFlag==0 
    %No regularization allowed if only one derivative is being forced, otherwise we may lose monotonicity
    regularizeFlag=0;
end
if nargin<5 || isempty(forceSign)
    %Determine if data is increasing or decreasing through the best 2-norm line fit:
    %pp=polyfit([1:numel(y)]',y,1);
    %forceSign=sign(pp(1));
    forceSign=sign(mean(diff(y))); %Alternative determination of increasing/decreasing
    %s=sign(median(diff(y)));
end

%%
if numel(y)~=length(y) %More than 1 vector (matrix input, acting along columns)
    z=nan(size(y));
    for i=1:size(y,2)
        z(:,i)=monoLS(y(:,i),p,monotonicDerivativeFlag,regularizeFlag);
    end
    
else %Vector input-data
    %Remove NaNs:   
    [y,idx]=removeNaN(y);

    [y,a,s]=flipIfNeeded(y,forceSign);

    %%Optimization
    %Get first guess:
    initDataGuess=defaultDataGuess(y); %Default is line: admissible as long as the slope is positive

    %Construct matrix that computes data from optimized variables: (By recursion!)
    [A,w0]=getMatrix(numel(y),monotonicDerivativeFlag,initDataGuess,regularizeFlag);

    %Now optimize:
    zz=optimize(A,y,w0,p);

    %Dealing with some ill-conditioned cases, in which a line is better than the solution found:
    if norm(zz-y,p)>norm(A*w0-y,p)
       zz=A*w0;
    end

    %Invert the flipping and positivization
    [zz]=unflip(zz,s,a);

    %Reconstructing data by adding the NaN values that were present
    z=nan(size(y)); 
    z(~idx)=zz;
end

end

function [f,g,h]=cost(y,A,w,p)
    f=norm(y-A*w,p)^p;
    g=p*sign(A*w-y)'.*abs(A*w-y)'.^(p-1) *A;
    h=p*(p-1)*A'*A;
end

function z=defaultDataGuess(y,order)
%if nargin<2 || order==0
    %First guess (init) for optimization target:
        x=[0:numel(y)-1]';
        pp=polyfit(x,y,1); %Fit a line to use as initial estimate: a line is always admissible!
        if pp(1)<0
            pp(1)=0;
            pp(2)=mean(y);
        end
        z=pp(2)+pp(1)*x;
%else
%    z=monoLS(y,2,order-1,[]);
%end
end

function [y,idx]=removeNaN(y)
    y=y(:); %Column vector
    idx=isnan(y);
    y=y(~idx);
end

function [z,a,s]=flipIfNeeded(y,s)
%Flips y as needed to get data that is positive, increasing and so on.

%To make it simple, we flip the data so that it is always increasing & positive
if s>0
    y=-y; %Data is now decreasing, f'<0
end
a=min(y)-1;
y=y-a; %Data is now f>0 & f'<0
z=flipud(y); %Data is now f>0, f'>0, as I inverted the 'x' axis
end

function [z]=unflip(y,s,a)
    %Invert the flipping and positivization
    z=flipud(y);
    z=z+a;
    if s>0
        z=-z;
    end
end

function zz=optimize(A,y,w0,p)
    if p==2
        %Solver 1: efficient but simple, does not converge for more than 3 derivatives
        %opts=optimset('Display','off');
        %w=lsqnonneg(A,y,opts);

        %Alternative solver: (this would allow us to pose the problem in different,
        %perhaps better conditioned, ways)
        %opts=optimoptions('quadprog','Display','off','Algorithm','trust-region-reflective');
        opts=optimoptions('quadprog','Display','off');
        B=A'*A;C=y'*A;
        w=quadprog(B,-C,[],[],[],[],zeros(size(w0)),[],w0,opts);
        
%         %Impose KKT conditions?: this improves the sol but is very slow,
%         %and doesnt get all the way to the optimum
%         d=(w'*B-C)'; %Gradient of the quadratic function with respect to w
%         %For each element, there are two options (if solution is optimal):
%         %1) d(i)>0 & w(i)=0, meaning the cost could decrease if w(i) decreases, but w(i) is at its lower bound
%         %2) d(i)=0 & w(i)>0[meaning optimal value of w(i) in an unconstrained sense]
%         %Note that w(i)<0 is inadmissible, and d(i)<0 means that w(i)=w(i)+dw is an admissible better solution
%         iter=0;
%         tol=1e-9/numel(y);
%         tol2=1e0*tol;
%         w(w<tol2)=0;
%         while any(w>tol2 & abs(d)>tol) && iter<1e5
%             dd=d.*(w>tol2).*(abs(d)>tol); %Projecting gradient along normal to admissibility set
%             H=dd'*B*dd /norm(dd)^2;
%             m=.1*norm(dd)/H;
%             idx2=w<m*dd;
%             dd(idx2)=0; %Not moving along directions where we would get w<0
%             w=w-m*dd;
%             w(idx2)=.5*w(idx2);
%            iter=iter+1;
%         end
        zz=A*w;
    else %Generic solver for other norms, which result in non-quadratic programs (solver is slower, but somewhat better)
        %As of Mar 07 2017, this didn't work properly. Convergence?
        opts=optimoptions('fmincon','Display','off','SpecifyObjectiveGradient',true);
        w1=fmincon(@(x) cost(y,A,x,p),w0,[],[],[],[],zeros(size(w0)),[],[],opts); 
        zz=A*w1;
    end
end

function [A,initSpaceGuess]=getMatrix(dataSize,order,initDataGuess,regularizeN)
    initSpaceGuess=[];
    %First, construct matrix that computes data from optimized variables: (By recursion!)
    A=tril(ones(dataSize));
    for i=1:order
        A(:,i+1:end)=cumsum(A(:,i+1:end),2,'reverse');
    end
    if regularizeN~=0 %Forcing the value of the m-th derivative (m=monotonicDerivativeFlag+1), 
       %which is the last constrained one, to be exactly 0 for the last n=regularizeFlag samples.
       %This avoids over-fitting to the first few datapoints (especially the
       %1st). %It is equivalent to reducing the size of the vector w() to be estimated.
       A(:,end-regularizeN+1:end)=[]; 
    end
    if nargin>2 && ~isempty(initDataGuess)
        w0=A\initDataGuess;
        initSpaceGuess=w0;
    end
end