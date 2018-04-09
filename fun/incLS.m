function [z] = incLS(y,normP,monotonicDerivativeN,regularizeN)
%This function does an p-norm minimization of (z-y), subject to z being non-decreasing. The returned vector z is a 'smoothed' version of y.
%%%%%INPUTS:
%y: column vector to smooth
%normP: norm used for minimization. Default normP=2 (least squares)
%monotonicDerivativeN: number of derivatives forced to be non-decreasing
% =1 means monotonic z & monotonic derivative (i.e. concave) 
% =2 forces the second derivative to be monotonic
%regularizeN: number of samples that are force to be =0 for
%the last derivative forced by monotonicDerivativeFlag. These samples are
%taken at the end of the z. It avoids overfitting of said samples.
%%%%%%OUTPUT:
%z=Best-fit approximation of data given constraints
%Notice that a monotonic best-fit is always piece-wise constant (derivative is null almost everywhere) in presence
%of noise (i.e. if data is not truly monotonic). In the same way, if more
%derivatives are enforced, the last enforced one will be piece-wise
%constant, so one order less will be piece-wise linear, two orders less
%will be piece-wise quadratic and so forth.

%Same as monoLS, but assumes that ALL derivatives are forced to be
%non-negative. Only takes vector as argument (NOT matrix)

%TODO: if monotonicDerivativeN==0, then enforce regularization on BOTH
%edges of the data (problem is symmetric and can be overfit at both ends)
%TODO:
%1) Confim that the optimization (p=2) works & is fast (against some good optimization library?)
%4) Add optimization over both increasing and decreasing functions, and
%choose the best fit (still according to p-norm) over both.
%5) Fix convergence for p~=2 (?)
%% ARGUMENT CHECK:
if nargin<2 || isempty(normP)
    normP=2;
end
if nargin<3 || isempty(monotonicDerivativeN)
    monotonicDerivativeN=0;
elseif monotonicDerivativeN>numel(y)
    error('Cannot force the sign of so many derivatives!')
end
if nargin<4 || isempty(regularizeN) || monotonicDerivativeN==0 
    %No regularization allowed if only one derivative is being forced, otherwise we may lose monotonicity
    regularizeN=0;
end

%% DO THE THING
    %Remove NaNs:   
    [y2,idx]=removeNaN(y);

    %%Optimization
    %Get first guess:
    initDataGuess=defaultDataGuess(y2); %Default is line: admissible as long as the slope is non-neg

    %Construct matrix that computes data from optimized variables: (By recursion!)
    [A,w0]=getMatrix(numel(y2),monotonicDerivativeN,initDataGuess,regularizeN);

    %Now optimize:
    zz=optimize(A,y2,w0,normP);

    %Dealing with some ill-conditioned cases, in which a line is better than the solution found:
    if norm(zz-y2,normP)>norm(A*w0-y2,normP)
       zz=A*w0;
    end

    %Reconstructing data by adding the NaN values that were present
    z=nan(size(y)); 
    z(~idx)=zz;
end

%% AUXILIARY FUNCTIONS:
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